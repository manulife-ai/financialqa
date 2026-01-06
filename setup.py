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
            "openai",
            "azure-ai-ml",
            "azure.ai.documentintelligence",
            "azure-core",
            "azure-identity",
            "azure-keyvault",
            "azure-search",
            "azure-search-documents",
            "python-dotenv",
            "langchain==0.3.27",
            "langchain-core==0.3.81",
            "langchain-openai==0.3.17",
            "langchain-xai==1.2.1",
            "langchain-community",
            "tiktoken",
            "transformers",
            "pandas",
            "numpy",
            "torch",
            "pypdf2",
            "pdf2image",
        ],
        package_dir={"": "src"},
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
