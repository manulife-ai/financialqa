import os
import sys
sys.path.append('..')
import argparse

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


from .helper import *
# from financial_qabot_table_reader.src.table2json_copy import extract_tables
from .parser import parse_pdfs, page_text_and_tables
# from parser import parse_pdfs, page_text_and_tables

from dotenv import load_dotenv
load_dotenv() # load environment variables from .env

# openai.api_type='azure'
# openai.api_version='2023-05-15'
# openai.api_base='https://use-gaa-openai-test1.openai.azure.com/'
# openai.api_key=os.getenv('OPENAI_API_KEY')

import warnings # not recommended, to suppress langchain openai error
warnings.filterwarnings("ignore")

'''
To-do's:

- How to call pip package with arguments from bash?
- Add add_docs option for index_docs()
- Fix langchain OpenAI error
'''

class IngestionPipeline:

    def __init__(self, azure_storage_connection_string, azure_storage_container_name,
                 azure_search_service_name, azure_search_index_name, azure_search_key,
                 run_test_pdf=''):

        self.azure_storage_connection_string = azure_storage_connection_string
        self.azure_storage_container_name = azure_storage_container_name
        self.azure_search_service_name = azure_search_service_name
        self.azure_search_index_name = azure_search_index_name
        self.azure_search_key = azure_search_key
        self.azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"
        self.run_test_pdf = run_test_pdf

    def get_blob_container_client(self, azure_storage_connection_string, azure_storage_container_name):

        blob_service_client = \
            BlobServiceClient.from_connection_string(self.azure_storage_connection_string)
        
        container_client = \
            blob_service_client.get_container_client(self.azure_storage_container_name)
        
        return container_client


    def extract_blob_paths(self, container_client):

        list_of_blob_paths = []
        
        for blob in container_client.list_blobs():
            if self.run_test_pdf:
                if blob.name != self.run_test_pdf:
                    continue
            path = 'https://' + os.environ['AZURE_STORAGE_CONTAINER_ACCOUNT'] + \
                    '.blob.core.windows.net/' + \
                    os.environ['AZURE_STORAGE_CONTAINER_NAME'] + '/' + blob.name
            list_of_blob_paths.append(path)
        
        return list_of_blob_paths
    

    def convert_pages_to_table_docs(self, paged_text_and_tables, metadata_page_span=1):

        lang_doc_tables = []
        for i, report in enumerate(paged_text_and_tables):
            num_pages = max(list(report.keys()))
            for page_num, tables_and_text in report.items():
                for table in tables_and_text.get('tables'):
                    # print(table, '\n')
                    # continue
                    # print('Length of original text:', len(tables_and_text.get('text')))
                    # print(tables_and_text.get('text'), '\n')
                    # tables_and_text.get('text')[:] = \
                    #     [text for text in tables_and_text.get('text') if text not in table.values]
                    # print('Length of deduplicated text:', len(tables_and_text.get('text')))
                    # print(tables_and_text.get('text'))
                    # return
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

        # text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=0)
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        lang_doc_tables_chunks = text_splitter.split_documents(lang_doc_tables)
    
        return lang_doc_tables_chunks
    

    def get_search_client(self, azure_search_endpoint):

        azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"
        search_client = SearchIndexClient(azure_search_endpoint,
                                 AzureKeyCredential(self.azure_search_key))
        return search_client

    
    def get_embedding_model(self):
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

        try:
            search_client.get_index(os.environ['AZURE_AI_SEARCH_INDEX_NAME'])
        except:
            print('No existing index with name', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], \
                    'in search service', os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
            index_exists = 0
        else:
            print('Existing index', os.environ['AZURE_AI_SEARCH_INDEX_NAME'], 'in search service', \
                    os.environ['AZURE_AI_SEARCH_SERVICE_NAME'])
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

        # return acs_vector_store


    def ingest_pdfs(self, azure_storage_connection_string, azure_storage_container_name, create_new_index=True, 
                    add_docs=False):

        '''
        Main driver code
        '''

        print('Getting blob container client....')
        container_client = self.get_blob_container_client(
            self.azure_storage_connection_string, 
            self.azure_storage_container_name)

        print('Extracting blob paths....')
        blob_paths = self.extract_blob_paths(container_client)
        
        print('Parsing pdfs....')
        result_dicts = parse_pdfs(blob_paths)

        print('Paging text and tables....')
        paged_text_and_tables = page_text_and_tables(result_dicts)

        print('Converting pages to table docs....')
        lang_doc_tables = self.convert_pages_to_table_docs(paged_text_and_tables)

        print('Chunking table docs....')
        lang_doc_tables_chunks = self.chunk_docs(lang_doc_tables)

        print('Getting Azure AI Search client....')
        search_client = self.get_search_client(self.azure_search_endpoint)

        print('Getting embedding model...')
        embedding_model = self.get_embedding_model()

        self.index_docs(search_client, lang_doc_tables_chunks, embedding_model, create_new_index=True, add_docs=False)


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

    # from dotenv import load_dotenv
    # load_dotenv()

    # args = arg_parser.parse_args()

    ingestion_pipeline = IngestionPipeline(os.environ['AZURE_STORAGE_CONNECTION_STRING'],
                                           os.environ['AZURE_STORAGE_CONTAINER_NAME'],
                                           os.environ['AZURE_AI_SEARCH_SERVICE_NAME'],
                                           os.environ['AZURE_AI_SEARCH_INDEX_NAME'],
                                           os.environ['AZURE_AI_SEARCH_KEY'],
                                           )

    # def __init__(self, azure_storage_connection_string, azure_storage_container_name,
    #              azure_search_service_name, azure_search_index_name, azure_search_key):

    ingestion_pipeline.ingest_pdfs(os.environ['AZURE_STORAGE_CONNECTION_STRING'], os.environ['AZURE_STORAGE_CONTAINER_NAME'])