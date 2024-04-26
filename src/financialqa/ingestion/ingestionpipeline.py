import os
import sys
# sys.path.append('..')
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

    def __init__(self, run_test_pdf=''):
    
        self.azure_storage_connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        self.azure_storage_container_name = os.environ['AZURE_STORAGE_CONTAINER_NAME']
        self.azure_search_service_name = os.environ['AZURE_AI_SEARCH_SERVICE_NAME']
        self.azure_search_index_name = os.environ['AZURE_AI_SEARCH_INDEX_NAME']
        self.azure_search_key = os.environ['AZURE_AI_SEARCH_KEY']
        self.azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"

        self.run_test_pdf = run_test_pdf
        
        self.logger = logging.getLogger(__name__)

    def get_blob_container_client(self):
        """Instantiate Azure Blob Storage container client."""
        self.logger.info('Getting Azure Storage blob container client...')
        blob_service_client = \
            BlobServiceClient.from_connection_string(self.azure_storage_connection_string)
        
        container_client = \
            blob_service_client.get_container_client(self.azure_storage_container_name)
        
        return container_client


    def extract_report_contents(self, container_client):
        """Extract blob paths from Azure Blob Storage container."""
        # list_of_blob_paths = []
        report_contents = {}
        self.logger.info('Extracting report contents...')
        logging.disable(logging.WARNING)
        for blob in container_client.list_blobs():
            if self.run_test_pdf:
                if blob.name != self.run_test_pdf:
                    continue
            report_name = blob.name
            report_blob_path = 'https://' + os.environ['AZURE_STORAGE_CONTAINER_ACCOUNT'] \
                                + '.blob.core.windows.net/' + os.environ['AZURE_STORAGE_CONTAINER_NAME'] \
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
        self.logger.info('Converting pages to table documents...')
        for i, report in enumerate(paged_text_and_tables):
            num_pages = max(list(report.keys()))
            for page_num, tables_and_text in report.items():
                for table in tables_and_text.get('tables'):
                    metadata = preprocess_text(' '.join(tables_and_text.get('text')))
                    lang_doc_tables.append(
                        Document(
                            page_content=table.to_string(),
                            # page_content=str(table),
                            metadata={
                                'text': metadata, 
                                'page_num': page_num,
                                }
                            )
                        )

        return lang_doc_tables
    

    def chunk_docs(self, lang_doc_tables):
        """Chunk LangChain Documents representing extracted tables."""
        self.logger.info('Chunking documents...')
        # text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=0)
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        lang_doc_tables_chunks = text_splitter.split_documents(lang_doc_tables)
    
        return lang_doc_tables_chunks
    

    def get_search_client(self):
        """Instantiate Azure AI Search client."""
        self.logger.info('Getting search client...')
        azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"
        search_client = SearchIndexClient(azure_search_endpoint,
                                 AzureKeyCredential(self.azure_search_key))
        return search_client

    
    def get_embedding_model(self):
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

    def index_docs(self, search_client, lang_doc_tables_chunks, embedding_model, create_new_index=False, add_docs=False):

        self.logger.info('Uploading documents to Azure AI Search index...')
        try:
            logging.disable(logging.WARNING)
            search_client.get_index(os.environ['AZURE_AI_SEARCH_INDEX_NAME'])
            logging.disable(logging.NOTSET)

        except:
            self.logger.info(
                ' '.join([
                    'No existing index with name', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], \
                    'in search service', os.environ['AZURE_AI_SEARCH_SERVICE_NAME']])
                )
            #  print('No existing index with name', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], \
            #         'in search service', os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
            index_exists = 0

        else:
            self.logger.info(
                ' '.join([
                'Existing index', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], 
                'in search service', os.environ['AZURE_AI_SEARCH_SERVICE_NAME']])
            )
            # print('Existing index', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], 'in search service', \
            #         os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
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
                vector_search_dimensions=len(embedding_model("Text")),
                vector_search_configuration="default",
                ),
                SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,),
            ]

            if create_new_index:
                if index_exists:
                    print('Deleting existing index', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], 'in search service', \
                            os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
                    search_client.delete_index(os.environ['AZURE_AI_SEARCH_INDEX_NAME'])
                print('Creating new index', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], 'in search service', \
                        os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
                acs_vector_store = AzureSearch(
                    azure_search_endpoint=self.azure_search_endpoint,
                    azure_search_key=os.environ['AZURE_AI_SEARCH_KEY'],
                    index_name=os.environ['AZURE_AI_SEARCH_INDEX_NAME'],
                    embedding_function=embedding_model,
                    fields=fields,
                )

                import time
                t = time.time()
                print('Pushing documents to Azure vector store...')
                acs_vector_store.add_documents(documents=lang_doc_tables_chunks)
                print(len(lang_doc_tables_chunks), 'documents successfully indexed in', \
                        round(time.time() - t), 'seconds')
        logging.disable(logging.NOTSET)

        # return acs_vector_store

    def ingest_pdfs(self, create_new_index=True, add_docs=False):
        """Parse, chunk, and ingest in one method."""
        container_client = self.get_blob_container_client()
        blob_paths = self.extract_blob_paths(container_client)
        result_dicts = parse_pdfs(blob_paths)
        paged_text_and_tables = page_text_and_tables(result_dicts)
        lang_doc_tables = self.convert_pages_to_table_docs(paged_text_and_tables)
        lang_doc_tables_chunks = self.chunk_docs(lang_doc_tables)
        search_client = self.get_search_client()
        embedding_model = self.get_embedding_model()
        self.index_docs(
            search_client, lang_doc_tables_chunks, embedding_model,
            create_new_index=True, add_docs=False
            )

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

    ingestion_pipeline = IngestionPipeline(run_test_pdf='MFC_QPR_2023_Q4_EN.pdf')
    ingestion_pipeline.ingest_pdfs()