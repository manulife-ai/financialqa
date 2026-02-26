from setuptools import setup, find_packages

VERSION = "0.0.2" 
DESCRIPTION = "Financial Report PDF Parser and Indexer"
LONG_DESCRIPTION = """
    Parse text, tables, and charts from (financial report) PDFs and upload 
    them into an Azure AI Search index for (financial data) question-answering (QA) 
    with an LLM in a Retrieval Augmented Generation (RAG) framework.
    """

setup(
        name="financialqa", 
        version=VERSION,
        author="Messiah Ataey",
        author_email="Messiah_Ataey@manulife.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "openai==1.61.1",
            "azure-ai-ml==1.24.0",
            "azure.ai.documentintelligence==1.0.0",
            "azure-core==1.32.0",
            "azure-identity==1.19.0",
            "azure-keyvault==4.2.0",
            "azure-search==1.0.0b2",
            "azure-search-documents==11.5.2",
            "python-dotenv==1.0.1",
            "langchain==0.3.17",
            "langchain-openai==0.3.3",
            "langchain-community==0.3.16",
            "tiktoken==0.8.0",
            "transformers==4.48.2",
            "pandas==2.2.3",
            "numpy==1.26.4",
            "torch==2.6.0",
            "pypdf2==3.0.1",
            "pdf2image==1.17.0",
        ],
        package_dir={"": "src"},
        python_requires='>=3.9',
        keywords=[
            "python", "azure", "pdf parser",
            "text parser", "table parser",
            "chart parser", "chart extractor", 
            "pdf question-answering",
            "financial question-answering",
        ],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
