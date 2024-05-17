import os
import sys
sys.path.append('..')
import argparse
import logging
logging.basicConfig(    
    # filename=logfile,    
    level=logging.DEBUG,    
    format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",    
    datefmt="%H:%M:%S")   

import openai
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

from .helper import preprocess_text
from .parser import parse_pdfs, page_text_and_tables

from dotenv import load_dotenv
load_dotenv(override=True)

import warnings  # not recommended, to suppress langchain openai error
warnings.filterwarnings("ignore")

'''
To-do's:
- Add option to add documents to index_docs()
- Address langchain OpenAI error
'''

class IngestionPipeline:
    """
    A class which provides functionality to extract blob storage container contents,
    construct and chunk financial tabular Document objects, and uploading the documents
    to an Azure AI Search index for use in a Financial-QA RAG application.
    """
    
    def __init__(self):
    
        self.azure_storage_connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        self.azure_storage_container_name = os.environ['AZURE_STORAGE_CONTAINER_NAME']
        self.azure_search_service_name = os.environ['AZURE_AI_SEARCH_SERVICE_NAME']
        self.azure_search_index_name = os.environ['AZURE_AI_SEARCH_INDEX_NAME']
        self.azure_search_key = os.environ['AZURE_AI_SEARCH_KEY']
        self.azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"

        self.logger = logging.getLogger(__name__)
        self.blob_storage_container = self._get_blob_container_client()
        self.search_client = self._get_search_client()
        self.embedding_model = self._get_embedding_model()

    def extract_report_contents(self, select_files=[]):
        """Extract blob reports from Azure Blob Storage container."""
        report_contents = {}
        self.logger.info('Extracting report contents from Azure Blob Storage container...')
        logging.disable(logging.WARNING)
        for blob in self.blob_storage_container.list_blobs():
            if select_files and blob.name not in select_files:
                continue
            report_name = blob.name
            report_blob_path = 'https://' \
                                + os.environ['AZURE_STORAGE_CONTAINER_ACCOUNT'] \
                                + '.blob.core.windows.net/' \
                                + os.environ['AZURE_STORAGE_CONTAINER_NAME'] \
                                + '/' + report_name
            name_contents = report_name.split('_')
            company_name = name_contents[0]
            report_quarter = name_contents[-1].replace('.pdf', '')
            report_contents[report_name] = {
                'company_name': company_name, 
                'report_quarter': report_quarter,
                'report_blob_path': report_blob_path,
                }
        logging.disable(logging.NOTSET)
        return report_contents
    
    def convert_pages_to_table_docs(self, paged_text_and_tables, metadata_page_span=1):
        """Create LangChain Document objects from extracted tables and text."""
        lang_doc_tables = []
        self.logger.info('Converting pages to LangChain documents...')
        for i, report in enumerate(paged_text_and_tables):
            company_name = report.get('company_name')
            report_quarter = report.get('report_quarter')
            report_blob_path = report.get('report_blob_path')
            pages = report.get('pages')
            for page_num, page_content in pages.items():
                for table in page_content.get('tables'):
                    metadata = preprocess_text(' '.join(page_content.get('text')))
                    lang_doc_tables.append(
                        Document(
                            # page_content=table.to_string(),
                            page_content=str(table),
                            metadata={
                                'text': metadata, 
                                'page_num': page_num,
                                'company_name': company_name,
                                'report_quarter': report_quarter,
                                'report_blob_path': report_blob_path,
                                }
                            )
                        )
                    if page_content.get('title') is not None:
                        lang_doc_tables[-1].metadata['page_titles'] = ', '.join(page_content.get('title'))
                    else:
                        lang_doc_tables[-1].metadata['page_titles'] = ''
                    if page_content.get('pageHeader') is not None:
                        lang_doc_tables[-1].metadata['page_headers'] = ', '.join(page_content.get('pageHeader'))
                    else:
                        lang_doc_tables[-1].metadata['page_headers'] = ''
                    if page_content.get('sectionHeader') is not None:
                        lang_doc_tables[-1].metadata['section_headers'] = ', '.join(page_content.get('sectionHeader'))
                    else:
                        lang_doc_tables[-1].metadata['section_headers'] = ''
                    if page_content.get('pageFooter') is not None:
                        lang_doc_tables[-1].metadata['page_footers'] = ', '.join(page_content.get('pageFooter'))
                    else:
                        lang_doc_tables[-1].metadata['page_footers'] = ''
        return lang_doc_tables
    
    def chunk_docs(self, lang_doc_tables, chunk_size=400):
        """Chunk LangChain Documents representing extracted tables."""
        self.logger.info('Chunking LangChain Documents...')
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        lang_doc_tables_chunks = text_splitter.split_documents(lang_doc_tables)
        return lang_doc_tables_chunks
    
    def get_search_index(self, add_docs=None, overwrite_index=False):
        """Get a specific search index instance with an option to add more docs to it."""
        self.logger.info('Getting Azure AI Search index {0}'.format(self.azure_search_index_name))
        logging.disable(logging.WARNING)
        try:
            self.search_client.get_index(os.environ['AZURE_AI_SEARCH_INDEX_NAME'])
        except:
            print('No existing index with name {0} in search service {1}'.\
                    format(self.azure_search_index_name, self.azure_search_service_name))
            index_exists = 0
        else:
            print('Found existing index with name {0} in search service {1}'.\
                    format(self.azure_search_index_name, self.azure_search_service_name))
            index_exists = 1
        finally:
            # default field names see https://python.langchain.com/docs/integrations/vectorstores/azuresearch/
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=len(self.embedding_model("Text")),
                    vector_search_configuration="default",
                ),
                SearchableField(
                    name="metadata",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    filterable=True,
                ),
                SimpleField(
                    name="company_name",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="report_quarter",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="page_titles",
                    type=SearchFieldDataType.String,
                    # filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="page_headers",
                    type=SearchFieldDataType.String,
                    # filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="page_footers",
                    type=SearchFieldDataType.String,
                    # filterable=True,
                    searchable=True,
                ),
                SimpleField(
                    name="section_headers",
                    type=SearchFieldDataType.String,
                    # filterable=True,
                    searchable=True,
                ),
            ]
            if overwrite_index:
                if index_exists:
                    print('Overwriting existing index {0} in search service {1}'.\
                            format(self.azure_search_index_name, self.azure_search_service_name))
                    # logging.info(
                    #     ' '.join([
                    #         'Deleting existing index', self.azure_search_index_name, \
                    #         'in search service', self.azure_search_service_name])
                    #     )
                    self.search_client.delete_index(self.azure_search_index_name)
                else:
                    print('No existing index named {0} to delete in search service {1}'.\
                        format(self.azure_search_index_name, self.azure_search_service_name))
                # logging.info(
                #     ' '.join([
                #         'Creating new index', self.azure_search_index_name, \
                #         'in search service', self.azure_search_service_name])
                # )
                # logging.info(' '.join(['No existing index to delete']))
            acs_vector_store = AzureSearch(
                azure_search_endpoint=self.azure_search_endpoint,
                azure_search_key=self.azure_search_key,
                index_name=self.azure_search_index_name,
                embedding_function=self.embedding_model,
                fields=fields,
            )
            if add_docs is not None:
                import time
                t = time.time()
                print('Adding {0} new documents to {1}'.\
                    format(len(add_docs), self.azure_search_index_name))
                acs_vector_store.add_documents(documents=add_docs)
                print('{0} new documents successfully added to index {1} in {2}s'.\
                    format(len(add_docs), self.azure_search_index_name, round(time.time() - t)))
        logging.disable(logging.NOTSET)
        return acs_vector_store

    def ingest_pdfs(self, overwrite_index=False):
        """Parse, chunk, and ingest in one method."""
        report_contents = self.extract_report_contents()
        result_dicts = parse_pdfs(report_contents)
        paged_text_and_tables = page_text_and_tables(result_dicts)
        lang_doc_tables = self.convert_pages_to_table_docs(paged_text_and_tables)
        lang_doc_tables_chunks = self.chunk_docs(lang_doc_tables)
        self.get_search_index(add_docs=lang_doc_tables_chunks, overwrite_index=overwrite_index)

    def _get_blob_container_client(self):
        """Instantiate Azure Blob Storage container client."""
        self.logger.info('Getting Azure Blob Storage container client...')
        blob_service_client = \
            BlobServiceClient.from_connection_string(self.azure_storage_connection_string)
        container_client = \
            blob_service_client.get_container_client(self.azure_storage_container_name)
        return container_client

    def _get_search_client(self):
        """Instantiate Azure AI Search client."""
        self.logger.info('Getting Azure AI Search client...')
        azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"
        search_client = SearchIndexClient(azure_search_endpoint,
                                 AzureKeyCredential(self.azure_search_key))
        return search_client

    def _get_embedding_model(self):
        """Instantiate OpenAI embedding model."""
        self.logger.info('Getting OpenAI embedding model...')
        embeddings = OpenAIEmbeddings(
            deployment='text-embedding-ada-002-v2',
            openai_api_base=os.environ['OPENAI_API_BASE'],
            openai_api_type=os.environ['OPENAI_API_TYPE'],
            openai_api_key=os.environ['OPENAI_API_KEY'],
            openai_api_version=os.environ['OPENAI_API_VERSION'],
            # chunk_size = 1
            )
        embedding_model=embeddings.embed_query
        return embedding_model

if __name__ == '__main__':

    # arg_parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--azure_storage_connection_string",
    #     type=str,
    #     help="azure storage connection string",
    # )
    # parser.add_argument(
    #     "--azure_storage_container_name",
    #     type=str,
    #     help="azure storage container name",
    # )
    # parser.add_argument(
    #     "--azure_ai_search_service_name",
    #     type=str,
    #     help="azure search service name",
    # )
    # parser.add_argument(
    #     "--azure_search_index_name",
    #     type=str,
    #     help="azure search index name",
    # )
    # parser.add_argument(
    #     "--azure_search_key",
    #     type=str,
    #     help="azure search key",
    # )

    # args = arg_parser.parse_args()

    ingestion_pipeline = IngestionPipeline()
    ingestion_pipeline.ingest_pdfs()