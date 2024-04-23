import os
import time

from financialqa.ingestion.parser import parse_pdfs, page_text_and_tables
from financialqa.ingestion.helper import cleanup_whitespace
from financialqa.ingestion.ingestionpipeline import IngestionPipeline

from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':

    ingestion_pipeline = IngestionPipeline(run_test_pdf='MFC_QPR_2023_Q4_EN.pdf')

    # parses, chunks, and ingests all-in-one
    # ingestion_pipeline.ingest_pdfs()

    # or you can do it step-by-step
    print('Getting blob container client...')
    container_client = ingestion_pipeline.get_blob_container_client()
    print('Extracting blob paths...')
    blob_paths = ingestion_pipeline.extract_blob_paths(container_client)
    print('Parsing pdfs...')
    result_dicts = parse_pdfs(blob_paths)
    print('Paging text and tables...')
    paged_text_and_tables = page_text_and_tables(result_dicts)
    print('Converting pages to table docs...')
    lang_doc_tables = ingestion_pipeline.convert_pages_to_table_docs(paged_text_and_tables)
    print('Chunking table docs....')
    lang_doc_tables_chunks = ingestion_pipeline.chunk_docs(lang_doc_tables)
    print('Getting Azure AI Search client...')
    search_client = ingestion_pipeline.get_search_client()
    print('Getting embedding model...')
    embedding_model = ingestion_pipeline.get_embedding_model()
    print('Ingesting the chunked documents...')
    ingestion_pipeline.index_docs(search_client, lang_doc_tables_chunks, embedding_model, create_new_index=True, add_docs=False)