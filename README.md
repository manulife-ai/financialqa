# Financial Report PDF Table Indexer
This package provides functionality to parse, chunk, and index the tables and text from (financial report) PDF files into an Azure AI Search vector database for (financial data) tabular question-answering (QA) with an LLM in a Retrieval Augmented Generation (RAG) framework.

## Features

The features that come included in this package are:

* Effective parsing of tables and text in pdfs using AI Document Intelligence services and storing them as a nested dictionary of the form:
```python
parsed_table = {page_num: {'text': [extracted_text], 'tables': [extracted_tables]}
```
* Attaches metadata to the extracted tables stored as `pandas.DataFrame` objects based on surrounding text on the same page and chunks whole tables using `langchain` functionality
* Ingests the chunked tables into the Azure AI Search vector store

## Installation and Usage

The following services are used in this package:

* Azure Storage Container
* Azure AI Document Intelligence
* Azure AI Search
* Azure OpenAI Studio

__Note that the pdfs are assumed to be stored in an Azure Blob Storage container__.

Fill the required environment variables for the used services in a file named `.env` in your working directory so that it can be discovered by the `dotenv` package at runtime and loaded. The required environment variables are:

```bash
AZURE_STORAGE_CONTAINER_NAME=""
AZURE_STORAGE_CONTAINER_ACCOUNT=""
AZURE_STORAGE_CONNECTION_STRING=""

DOCUMENT_ENDPOINT=""
DOCUMENT_KEY=""

AZURE_AI_SEARCH_SERVICE_NAME=""
AZURE_AI_SEARCH_INDEX_NAME=""
AZURE_AI_SEARCH_KEY=""

OPENAI_API_BASE=""
OPENAI_API_KEY=""
OPENAI_API_VERSION=""
OPENAI_API_TYPE=""
```

__(coming soon...)__ Then, run the following to install the current release from the command line:
```bash
$ pip install financialqa
```
The package can then be imported and used in a script:
```python
import financialqa
from financialqa.ingestion.ingestionpipeline import IngestionPipeline

from dotenv import load_dotenv
load_dotenv()  # load environment variables

# Run a single test pdf through the ingestion pipeline
ingestion_pipeline = IngestionPipeline(run_test_pdf='MFC_QPR_2023_Q4_EN.pdf')

# Parse, chunk, and index extracted tables into Azure AI Search at once
ingestion_pipeline.ingest_pdfs() 
```

Or run directly from the command line:

```bash
$ python3 -m financialqa.ingestion.ingestionpipeline
```

# To-do's
* Apply PEP 8 style guide
* Attach section header metadata to ingested tables
* Include class docstrings
* Specify install_packages dependencies in setup
* Include more test cases
* Improve logging
* Include exception handling for API calls
* Upload final copy to Python Package Index