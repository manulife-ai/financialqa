from setuptools import setup, find_packages

VERSION = "0.0.1" 
DESCRIPTION = "Financial Report PDF Table Indexer"
LONG_DESCRIPTION = """
    Parse, chunk, and index the tables and text from 
    (financial report) PDF files into an Azure AI Search vector 
    database for (financial data) tabular question-answering (QA) with
    an LLM in a Retrieval Augmented Generation (RAG) framework.
    """

setup(
        name="financialqa", 
        version=VERSION,
        author="Messiah Ataey",
        author_email="Messiah_Ataey@manulife.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "openai==0.28.1",
            "azure-ai-ml",
            "azure.ai.formrecognizer",
            "langchain",
            "langchain-community==0.0.20",
            "azure-search",
            "azure-search-documents",
            "python-dotenv",
            "azure-core",
            "tiktoken",
        ],
        # packages=find_packages(),
        package_dir={"": "src"},
        keywords=[
            "python", "azure", "pdf parser",
            "pdf indexing", "table indexing", 
            "financial question-answering"
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
