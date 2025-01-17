import os
import sys
import json
import copy
import logging
import argparse

from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    VectorSearch,
    HnswParameters,
    VectorSearchProfile,
    VectorSearchAlgorithmKind,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmMetric,
)

from dotenv import load_dotenv
load_dotenv(override=True)

from .helper import preprocess_text
from .parser import parse_pdfs, page_pdf_contents

import warnings  # not recommended, to suppress langchain openai error
warnings.filterwarnings("ignore")

'''
To-do's:
    - Find alternative method to load PDF paths from Azure Blob storage with 
        newer DocumentIntelligenceClient class
'''

class IngestionPipeline:
    """
    A class which provides functionality to extract blob storage container contents,
    construct and chunk financial tabular Document objects, and uploading the documents
    to an Azure AI Search index for use in a Financial-QA RAG application.
    """
    
    def __init__(self):
        self.configure_logging()
        self._load_api_vars()
        self.blob_storage_container = self._get_blob_container_client()
        self.search_client = self._get_search_client()
        self.embedding_model = self._get_embedding_model()

    def configure_logging(self):
        logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')  
        logger.setLevel(logging.CRITICAL)
        httpx_logger = logging.getLogger('httpx')  
        httpx_logger.setLevel(logging.WARNING)  # Set the httpx logging level to WARNING to suppress Azure INFO logs
        self.logger = logging.getLogger(__name__)  
        logging.basicConfig(        
            level=logging.WARNING,    
            format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",    
            datefmt="%H:%M:%S",
            force=True,
        )  
        self.logger.info('Configured logging.')

    def extract_report_contents(self, load_from_local=False):
        """Extract blob reports from Azure Blob Storage container."""
        report_contents = {}
        self.logger.info(
            "Extracting {0} PDF contents from Azure Blob Storage container '{1}'...".\
                format(
                    len(select_files),
                    self.azure_storage_container_name,
                )
        )
        for blob in self.blob_storage_container.list_blobs():
            if select_files and blob.name not in select_files:
                continue
            report_name = blob.name
            self.logger.info("Extracting contents from file '{0}'.".format(report_name))
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
        return report_contents
    
    def add_chartvlm_docs(
            self, 
            chartvlm_output_filepath, 
            company_list
        ):
        with open(chartvlm_output_filepath, "r") as file:
            chartvlm_results = json.load(file)
        chartvlm_docs = []
        for chart_filename, chartvlm_result in chartvlm_results.items():
            company_name = chart_filename.split('/')[-1].split('_')[0]
            if company_name in company_list:
                company_name = company_name.upper()
                if company_name.lower() == 'manulife':
                    company_name = 'MFC'
                chart_title = chartvlm_result.get('title')
                chartvlm_table_json = chartvlm_result.get('json')
                chartvlm_docs.append(
                    Document(
                        page_content=str(chartvlm_result),
                        metadata={
                            'text': chart_title.strip(), 
                            'page_num': 'N/A',
                            'company_name': company_name,
                            'report_quarter': 'N/A',
                            'report_blob_path': 'N/A',
                            'page_titles': 'N/A',
                            'page_headers': 'N/A',
                            'section_headers': 'N/A',
                            'page_footers': 'N/A',
                            }
                        )
                    )
        return chartvlm_docs
    
    def crop_images_from_pdfs(
            self, 
            pdf_pages_dict, 
            save_as_jpg=False,
            overwrite_pdf_images=False,
            overwrite_jpg_images=False,
        ):
        """
        Crop and save images from PDFs using PyPDF2 with bounding box coordinates
        generated by Azure Document Intelligence.
        """
        for pdf_name, pdf_contents in pdf_pages_dict.items():
            pdf_pages = pdf_contents.get('pages')
            for page_num, page_dict in pdf_pages.items():
                if not 'figures' in page_dict:
                    continue
                print(f'Processing figures on page {page_num} of {pdf_name}...')
                pdf_figures = page_dict.get('figures')
                pdf_file = os.path.join('../data', pdf_name + '.pdf')
                reader = PdfReader(pdf_file)
                page = reader.pages[page_num-1]
                for figure_name, figure_items in pdf_figures.items():
                    bounding_regions_polygon = figure_items.get('bounding_regions_polygon')
                    # found this 72 scaling factor here:
                    # https://github.com/microsoft/Form-Recognizer-Toolkit/blob/main/SampleCode/Python/sample_figure_understanding.ipynb
                    bounding_regions_polygon = [72 * polygon for polygon in bounding_regions_polygon]
                    x1, y1, x2, y2, x3, y3, x4, y4 = bounding_regions_polygon
                    page_mediabox_upper_left = copy.deepcopy(page.mediabox.upper_left)
                    page_mediabox_lower_right = copy.deepcopy(page.mediabox.lower_right)
                    new_upper_left = (
                        page.mediabox.upper_left[0].as_numeric() + x1, 
                        page.mediabox.upper_left[1].as_numeric() - y1
                    )
                    new_upper_right = (
                        page.mediabox.upper_left[0].as_numeric() + x2, 
                        page.mediabox.upper_left[1].as_numeric() - y2
                    )
                    new_lower_right = (
                        page.mediabox.upper_left[0].as_numeric() + x3, 
                        page.mediabox.upper_left[1].as_numeric() - y3
                    )
                    new_lower_left = (
                        page.mediabox.upper_left[0].as_numeric() + x4, 
                        page.mediabox.upper_left[1].as_numeric() - y4
                    )
                    page.mediabox.upper_left = new_upper_left
                    page.mediabox.lower_right = new_lower_right
                    # page.mediabox.upper_right = new_upper_right
                    # page.mediabox.lower_left = new_lower_left
                    writer = PdfWriter() # find a workaround to reinitalizing
                    writer.add_page(page)
                    page.mediabox.upper_left = page_mediabox_upper_left
                    page.mediabox.lower_right = page_mediabox_lower_right
                    output_dir = os.path.join('../data/outputs', pdf_name)
                    if not os.path.isdir(output_dir):
                        print(f"Creating a directory '{output_dir}' to save outputs...")
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, figure_name + '.pdf')
                    if overwrite_pdf_images:
                        with open(output_path, 'wb') as out_f:
                            writer.write(out_f)
                    if save_as_jpg:
                        jpg_output_folder = os.path.join(
                            '../data/outputs', 
                            pdf_name, 
                            'jpg_outputs'
                        )
                        if not os.path.isdir(jpg_output_folder):   
                            print(f"Creating a directory '{jpg_output_folder}' to save JPEG outputs...")
                            os.makedirs(jpg_output_folder)
                        images = convert_from_path(
                            output_path,
                            fmt='JPEG',
                            output_folder=jpg_output_folder,
                        )
                        if overwrite_jpg_images:
                            for img in images:
                                jpg_file_path = os.path.join(jpg_output_folder, figure_name + '.jpg')
                                print(f"Saving PDF '{pdf_name}' as JPEG to folder '{jpg_output_folder}'.")
                                img.save(fp=jpg_file_path)

    def convert_paged_pdf_contents_to_docs(
            self, 
            paged_pdf_contents,
            convert_tables=True,
            convert_charts=True,
        ):
        """Create LangChain Document objects from extracted PDF contents."""
        lang_doc_text = []
        lang_doc_tables = []
        lang_doc_charts = []
        self.logger.info('Converting extracted PDF page contents to LangChain Documents...')
        for pdf_name, pdf_items in paged_pdf_contents.items():
            company_name = pdf_items.get('company_name')
            report_quarter = pdf_items.get('report_quarter')
            pages = pdf_items.get('pages')
            for page_num, page_content in pages.items():
                # Convert text content by default
                text_content = preprocess_text(' '.join(page_content.get('text')))
                lang_doc_text.append(
                    Document(
                        page_content=text_content,
                        metadata={
                            'page_num': page_num,
                            'pdf_name': pdf_name,
                            'company_name': company_name,
                            'report_quarter': report_quarter,
                            'page_titles': '',
                            'page_headers': '',
                            'section_headers': '',
                            'page_footers': '',
                            # 'report_blob_path': report_blob_path,
                            }
                        )
                    )
                if convert_tables:
                    for table in page_content.get('tables'):
                        lang_doc_tables.append(
                            Document(
                                page_content=str(table),
                                metadata={
                                    'text': text_content, 
                                    'pdf_name': pdf_name,
                                    'page_num': page_num,
                                    'company_name': company_name,
                                    'report_quarter': report_quarter,
                                    'page_titles': '',
                                    'page_headers': '',
                                    'section_headers': '',
                                    'page_footers': '',
                                    # 'report_blob_path': report_blob_path,
                                    }
                                )
                            )
                if convert_charts:
                    pass 
            if page_content.get('title') is not None:
                lang_doc_text[-1].metadata['page_titles'] = ', '.join(page_content.get('title'))
                lang_doc_tables[-1].metadata['page_titles'] = ', '.join(page_content.get('title'))
            if page_content.get('pageHeader') is not None:
                lang_doc_text[-1].metadata['page_headers'] = ', '.join(page_content.get('pageHeader'))
                lang_doc_tables[-1].metadata['page_headers'] = ', '.join(page_content.get('pageHeader'))
            if page_content.get('sectionHeader') is not None:
                lang_doc_text[-1].metadata['section_headers'] = ', '.join(page_content.get('sectionHeader'))
                lang_doc_tables[-1].metadata['section_headers'] = ', '.join(page_content.get('sectionHeader'))
            if page_content.get('pageFooter') is not None:
                lang_doc_text[-1].metadata['page_footers'] = ', '.join(page_content.get('pageFooter'))
                lang_doc_tables[-1].metadata['page_footers'] = ', '.join(page_content.get('pageFooter'))
        self.logger.info("Created {0} text Documents.".format(len(lang_doc_text)))
        self.logger.info("Created {0} table Documents.".format(len(lang_doc_tables)))
        self.logger.info("Created {0} chart Documents.".format(len(lang_doc_charts)))
        return lang_doc_text, lang_doc_tables, lang_doc_charts
    
    def chunk_docs(self, lang_docs, chunk_size=400):
        """Chunk list of Document objects."""
        self.logger.info(
            'Chunking {0} Documents with a token chunk size of {1}...'.\
                format(len(lang_docs), 
                chunk_size
                )
            )
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        lang_doc_chunks = text_splitter.split_documents(lang_docs)
        self.logger.info("Chunked {0} Documents to {1}.".format(
            len(lang_docs), len(lang_doc_chunks))
        )
        return lang_doc_chunks
    
    def get_search_index(self, add_docs=None, overwrite_index=False):
        """Get a specific search index instance with an option to add more Documents to it."""
        self.logger.info("Getting Azure AI Search index '{0}'...".format(self.azure_search_index_name))
        try:
            self.search_client.get_index(self.azure_search_index_name)
        except:
            self.logger.info("Did not find existing index '{0}' in search service {1}.".\
                    format(self.azure_search_index_name, self.azure_search_service_name))
            index_exists = 0
        else:
            self.logger.info("Found existing index '{0}' in search service {1}.".\
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
                    vector_search_dimensions=len(
                        self.embedding_model.embed_query("Text")
                    ),
                    # vector_search_configuration="default",
                    vector_search_profile_name="finqaHnsw",
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
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="finqaHnsw",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE,
                        ),
                    ),
                    # ExhaustiveKnnAlgorithmConfiguration(
                    #     name="default_exhaustive_knn",
                    #     kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    #     parameters=ExhaustiveKnnParameters(
                    #         metric=VectorSearchAlgorithmMetric.COSINE
                    #     ),
                    # ),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="finqaHnsw",
                        algorithm_configuration_name="finqaHnsw",
                    ),
                    # VectorSearchProfile(
                    #     name="myExhaustiveKnnProfile",
                    #     algorithm_configuration_name="default_exhaustive_knn",
                    # ),
                ],
            )
            if overwrite_index:
                if index_exists:
                    self.logger.info("Found an existing index named '{0}' in search service '{1}' to overwrite.".
                        format(self.azure_search_index_name, self.azure_search_service_name))
                    self.search_client.delete_index(self.azure_search_index_name)
                else:
                    self.logger.info("Did not find an existing index named '{0}' in search service {1} to overwrite.".
                        format(self.azure_search_index_name, self.azure_search_service_name))
            acs_vector_store = AzureSearch(
                azure_search_endpoint=self.azure_search_endpoint,
                azure_search_key=self.azure_search_key,
                index_name=self.azure_search_index_name,
                embedding_function=self.embedding_model.embed_query,
                fields=fields,
                additional_search_client_options={"retry_total": 4},
                vector_search=vector_search,
            )
            if add_docs is not None:
                import time
                t = time.time()
                self.logger.info("Attempting to add {0} Documents to index '{1}'...".format(
                    len(add_docs), self.azure_search_index_name)
                )
                acs_vector_store.add_documents(documents=add_docs)
                self.logger.info("A total of {0} Documents were successfully added to index '{1}' in {2}s.".\
                    format(len(add_docs), self.azure_search_index_name, round(time.time() - t))
                )
        return acs_vector_store

    def ingest_pdfs(
            self, 
            select_files=[], 
            overwrite_index=False,
        ):
        """Extract, parse, chunk, and index PDFs in a single function call."""
        report_contents = self.extract_report_contents(select_files)
        result_dicts = parse_pdfs(report_contents)
        paged_text_and_tables = page_pdf_contents(result_dicts)
        lang_doc_tables = self.convert_paged_pdf_contents_to_docs(paged_text_and_tables)
        lang_doc_tables_chunks = self.chunk_docs(lang_doc_tables)
        self.get_search_index(
            add_docs=lang_doc_tables_chunks, 
            overwrite_index=overwrite_index
        )
        # vector_store = self.get_search_index(add_docs=lang_doc_tables_chunks, overwrite_index=overwrite_index)
        # return vector_store

    def _get_blob_container_client(self):
        """Instantiate Azure Blob Storage container client."""
        self.logger.info("Getting Azure Blob Storage container client '{0}'...".format(self.azure_storage_container_name))
        blob_service_client = \
            BlobServiceClient.from_connection_string(self.azure_storage_connection_string)
        container_client = \
            blob_service_client.get_container_client(self.azure_storage_container_name)
        return container_client

    def _get_search_client(self):
        """Instantiate Azure AI Search client."""
        self.logger.info("Getting Azure AI Search client from service '{0}'...".format(self.azure_search_service_name))
        azure_search_endpoint = "https://" + self.azure_search_service_name + ".search.windows.net"
        search_client = SearchIndexClient(azure_search_endpoint, AzureKeyCredential(self.azure_search_key))
        return search_client

    def _get_embedding_model(self):
        """Instantiate OpenAI embedding model."""
        openai_api_deployment = "text-embedding-3-small"
        self.logger.info("Getting OpenAI embedding model {0}'...".format(openai_api_deployment))
        embeddings = AzureOpenAIEmbeddings(
            deployment=openai_api_deployment,
            azure_endpoint=os.environ['AZURE_ENDPOINT'],
            openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            # openai_api_key=os.environ['OPENAI_API_KEY'],
            show_progress_bar=True,
            chunk_size = 1
        )
        return embeddings
    
    def _load_api_vars(self):
        self.azure_search_key = os.getenv('AZURE_AI_SEARCH_KEY')
        self.azure_search_index_name = os.getenv('AZURE_AI_SEARCH_INDEX_NAME')
        self.azure_search_service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')
        self.azure_storage_container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
        self.azure_storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.azure_search_endpoint = \
            "https://" + self.azure_search_service_name + ".search.windows.net"
        self.logger.info('Set API variables.')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--select_files",
    #     type=str,
    #     help="Specific files to select in index.",
    # )
    # args = parser.parse_args()
    ingestion_pipeline = IngestionPipeline()
    select_files = [
    ]
    ingestion_pipeline.ingest_pdfs(
        select_files, overwrite_index=True
    )