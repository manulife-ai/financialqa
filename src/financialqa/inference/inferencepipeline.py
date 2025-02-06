import os
import logging
import argparse

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(    
    # filename=logfile,    
    level=logging.INFO,    
    format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",    
    datefmt="%H:%M:%S"
)   
logger = logging.getLogger(__name__)

"""
To-do's:
- See https://github.com/langchain-ai/langchain/issues/27511 for program termination
    error issue with AzureSearch
"""

class InferencePipeline:
    """
    Ask questions to an LLM about multi-structured PDF contents retrieved from an index.
    
    Attributes:
        chat_model (AzureChatOpenAI): The LLM used for completions.
        embedding_model (AzureOpenAIEmbeddings): The model used for generating 
            vector embeddings.
        ai_search_index (AzureSearch): The Azure AI Search index.
    """

    def __init__(self):
        self._load_api_vars()
        self.chat_model = self._get_azure_chat_model()
        self.embedding_model = self._get_embedding_model()
        self.ai_search_index = self._get_ai_search_index()

    def invoke_llm(self, query, retrieved_chunks): 
        """
        Invoke the LLM with a given query and retrieved context.
        
        Inputs:
            query (str): Query for RAG pipeline.
            retrieved_chunks (list): Retrieved context from index.

        Outputs:
            llm_response (str): Response from LLM.
        """
        messages = [
            (
                "system",
                f"""
                You are a helpful assistant that answers questions \
                based on a given input.

                Question:
                {query}

                Input:
                {retrieved_chunks}
                """
            ),
        ]
        llm_response = self.chat_model.invoke(messages)
        return llm_response

    def query_index(
            self, 
            query, 
            company_name='',
            top_k=3, 
        ):
        """
        Query the index for Documents based on a similarity search.
        
        Inputs:
            query (str): Query used to search against index.
            company_name (str): Company name used to filter index.
            top_k (int): Top number of Documents to retrieve from index (default is 3).
        
        Outputs:
            retrieved_chunks (list): Retrieved Document objects from index.
        """
        filters = ''
        if company_name:
            filters = f"company_name eq '{company_name}'"
        retrieved_chunks = self.ai_search_index.similarity_search(
            query=query,
            k=top_k,
            filters=filters,
            search_type="similarity",
        )
        return retrieved_chunks
    
    def invoke_model(
            self, 
            query,
            company_name='',
            top_k=3,
        ):
        """
        Invoke the RAG model based on a given query, optional company name filter, 
        and top-k Documents to return from the index.
        
        Inputs:
            query (str): Query for RAG pipeline.
            company_name (str): Company name used to filter index.
            top_k (int): Top number of Documents to retrieve from index (default is 3).
        
        Outputs:
            None
        """
        retrieved_chunks = self.query_index(
            query, 
            top_k=top_k,
            company_name=company_name,
        )
        logger.info(f'Retrieved context: {retrieved_chunks}')
        llm_response = self.invoke_llm(
            query, 
            retrieved_chunks
        ).dict()
        logger.info(
            "{0} response: {1}".format(
                os.getenv('AZURE_OPENAI_COMPLETION_MODEL'),
                llm_response.get('content'),
        ))
        
    def _get_azure_chat_model(self):
        """
        Instantiate Azure OpenAI chat model.
        
        Returns:
            AzureChatOpenAI: Azure OpenAI chat model.
        """
        logger.info(
            "Getting Azure OpenAI completion model '{0}' from endpoint {1}..".format(
                self.azure_openai_completion_model,
                self.azure_openai_endpoint,
            ))
        chat_model = AzureChatOpenAI(
            azure_deployment=self.azure_openai_completion_model,
            azure_endpoint=self.azure_openai_endpoint,
            openai_api_version=self.azure_openai_version,
            openai_api_key=self.azure_openai_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return chat_model

    def _get_embedding_model(self):
        """
        Instantiate Azure OpenAI embedding model.

        Returns:
            AzureOpenAIEmbeddings: The model used for generating vector embeddings.
        """
        logger.info(
            "Getting Azure OpenAI embedding model '{0}' from endpoint {1}..".format(
                self.azure_openai_embedding_model,
                self.azure_openai_endpoint,
            ))
        embeddings = AzureOpenAIEmbeddings(
            deployment=self.azure_openai_embedding_model,
            azure_endpoint=self.azure_openai_endpoint,
            openai_api_version=self.azure_openai_version,
            openai_api_key=self.azure_openai_key,
        )
        return embeddings

    def _get_ai_search_index(self):
        """
        Instantiate Azure AI Search index.

        Returns:
            AzureSearch: Azure AI Search index object.
        """
        logger.info("Getting Azure AI Search index '{0}' from service '{1}'..".format(
            self.azure_search_index_name,
            self.azure_search_service_name,
            ))
        ai_search_index = AzureSearch(
            azure_search_endpoint=self.azure_search_endpoint,
            azure_search_key=self.azure_search_key,
            index_name=self.azure_search_index_name,
            embedding_function=self.embedding_model.embed_query,
        )
        return ai_search_index

    def _load_api_vars(self):
        """
        Load API variables from environment variables.

        Returns:
            None
        """
        self.azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
        self.azure_openai_type = os.getenv('AZURE_OPENAI_TYPE')
        self.azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_embedding_model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')
        self.azure_openai_completion_model = os.getenv('AZURE_OPENAI_COMPLETION_MODEL')
        self.azure_search_key = os.getenv('AZURE_AI_SEARCH_KEY')
        self.azure_search_index_name = os.getenv('AZURE_AI_SEARCH_INDEX_NAME')
        self.azure_search_service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')
        self.azure_search_endpoint = \
            "https://" + self.azure_search_service_name + ".search.windows.net"
        logger.info("Loaded API variables")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query for the Financial QA model",
    )
    parser.add_argument(
        "--company_name",
        type=str,
        help="Company name for filtering the index",
        default=3,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Top number of Documents based on query to retrieve from index",
        default=3,
    )
    args = parser.parse_args()
    inferencepipeline = InferencePipeline()
    inferencepipeline.invoke_model(
        query=args.query,
        company_name=args.company_name,
        top_k=args.top_k
    )