# Financial Report pdf Indexer
This package provides functionality to parse, chunk, and index the tables and text from (financial report) pdfs into an Azure AI Search vector database for use in a Retrieval Augmented Generation question-answering framework for (financial data) tabular understanding.

## Features

The features that come included in this package are:

* Effective parsing of tables and text in pdfs using AI Document Intelligence services and storing them as a nested dictionary of the form:
```python
parsed_table = {page_num: {'text': [extracted_text], 'tables': [extracted_tables]}
``` 
* Attaches metadata to the extracted tables based on surrounding text in the same page and then chunks whole tables stored as DataFrame objects using langchain tools
* Ingests the chunked tables into the Azure AI Search vector store.

## How To Use

First, you will need to setup the necessary configuration variables required to access the services used in this package, namely:

* Azure AI Document Intelligence
* Azure AI Search
* Azure Storage Container
* OpenAI API

Please fill the required environment variables in the `template.env` file and then rename and save the file as `.env` so that it can be found by the `dotenv` package at runtime. __Note that the pdfs are assumed to be located in an Azure Blob Storage container__, the details of which are provided in  ```template.env```, namely the container account name, container name, and the storage connection string.

To install the current release, from  your command line:

```bash
$ pip install financial-pdf-indexer
```
Then, you can run it directly from the command line:
```bash
python3 -m ingestionpipeline
```
or you can import the package as a module and use it directly in your code:
```python
import financialqa
```

# To-do's
* Fix module imports
* Include exception handling for invalid environment variables