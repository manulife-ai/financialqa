# Financial Report PDF Indexing
Parse tables and text from (financial report) PDFs and upload them into an Azure AI Search index for (financial data) tabular question-answering (QA) with an LLM in a Retrieval Augmented Generation (RAG) framework.

## Features
* Effective parsing of tables and text from PDFs stored in an Azure Blob Storage container using Azure Document Intelligence
<!-- ```python
{page_num: {'text': [extracted_text], 'tables': [extracted_tables]}
``` -->
* Document object creation with surrounding text metadata and chunking using ```LangChain``` functionality
* Document embedding generation using Azure OpenAI and insertion into an Azure AI Search index

![Features](ingestion.PNG)

## Services Used
* Azure Blob Storage Container
* Azure Document Intelligence
* Azure AI Search
* Azure OpenAI Studio

## Usage
First, clone and navigate to the repository:
```bash
git clone https://github.com/manulife-gft/FinancialQA
cd FinancialQA/
```
Then, install the package locally using ```pip```:
```bash
pip install -e .
```
<!-- __(coming soon...)__ Then, run the following to install the current release from the command line:
```bash
$ pip install financialqa
``` -->

Set the required environment variables shown below in a file named `.env` (in the ```FinancialQA/``` directory), so that it can be discovered and loaded by the `dotenv` package at runtime. The targeted PDF files must be hosted in an Azure Blob Storage container, as specified by the ```AZURE_STORAGE_*``` variables. The ```AZURE_AI_SEARCH_INDEX_NAME``` variable can be set to any name.

```bash
AZURE_STORAGE_CONTAINER_NAME=""
AZURE_STORAGE_CONTAINER_ACCOUNT=""
AZURE_STORAGE_CONNECTION_STRING=""

DOCUMENT_INTEL_KEY=""
DOCUMENT_INTEL_ENDPOINT=""

AZURE_AI_SEARCH_KEY=""
AZURE_AI_SEARCH_INDEX_NAME=""
AZURE_AI_SEARCH_SERVICE_NAME=""

OPENAI_API_KEY=""
OPENAI_API_BASE=""
OPENAI_API_TYPE=""
OPENAI_API_VERSION=""
```
The ```financialqa``` package can then be imported and used in a script:
```python
from financialqa.ingestion.ingestionpipeline import IngestionPipeline

ingestion_pipeline = IngestionPipeline()

# Extract, parse, chunk, and index tables into Azure AI Search
ingestion_pipeline.ingest_pdfs() 
```

Or run directly from the command line:

```bash
$ python3 -m financialqa.ingestion.ingestionpipeline
```

## To-do's
* Implement batch uploading to index
* Provide configurable parameters
* Add proper class/method docstrings
* Add unittest test cases
* Add exception handling for API calls
* Upload final copy to Python Package Index