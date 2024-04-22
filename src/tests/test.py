import os
from financialqa.ingestion.ingestionpipeline import IngestionPipeline
from financialqa.ingestion.helper import cleanup_whitespace

# from dotenv import load_dotenv
# load_dotenv()

# ingestion_pipeline = IngestionPipeline(os.environ['AZURE_STORAGE_CONNECTION_STRING'],
#                                         os.environ['AZURE_STORAGE_CONTAINER_NAME'],
#                                         os.environ['AZURE_AI_SEARCH_SERVICE_NAME'],
#                                         os.environ['AZURE_AI_SEARCH_INDEX_NAME'],
#                                         os.environ['AZURE_AI_SEARCH_KEY'],
#                                         run_test_pdf='MFC_QPR_2023_Q4_EN.pdf'
#                                         )

ingestion_pipeline = IngestionPipeline(run_test_pdf='MFC_QPR_2023_Q4_EN.pdf')

# def __init__(self, azure_storage_connection_string, azure_storage_container_name,
#              azure_search_service_name, azure_search_index_name, azure_search_key):

# ingestion_pipeline.ingest_pdfs(os.environ['AZURE_STORAGE_CONNECTION_STRING'], os.environ['AZURE_STORAGE_CONTAINER_NAME'])

ingestion_pipeline.ingest_pdfs()