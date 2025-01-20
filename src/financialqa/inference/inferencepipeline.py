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

# logging.basicConfig(    
#     # filename=logfile,    
#     level=logging.INFO,    
#     format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",    
#     datefmt="%H:%M:%S"
# )   
# logger = logging.getLogger(__name__)

"""
To-do's:
- See https://github.com/langchain-ai/langchain/issues/27511 for program termination
    error issue with AzureSearch
"""

class InferencePipeline:
    """
    """
    def __init__(self):
        self._load_api_vars()
        self.chat_model = self._get_azure_chat_model()
        self.embedding_model = self._get_embedding_model()
        self.ai_search_index = self._get_ai_search_index()

    def invoke_model(self, query, retrieved_chunks): 
        messages = [
            (
                "system",
                f"""
                You are a helpful assistant that answers questions based on a given input.

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
            top_k=3, 
            search_type='similarity',
            company_name='',
        ):
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
    
    def infer(
            self, 
            query,
            company_name='',
            top_k=3,
    ):
        retrieved_chunks = self.query_index(
            query, 
            top_k=top_k,
            company_name=company_name,
        )
        print(f'Retrieved context: {retrieved_chunks}')
        llm_response = self.invoke_model(
            query, 
            retrieved_chunks
        ).dict()
        print(
            f"{os.environ['AZURE_OPENAI_COMPLETION_MODEL']} response: {llm_response.get('content')}"
        )
        
    def _get_embedding_model(self):
        """Instantiate OpenAI embedding model."""
        openai_api_deployment = "text-embedding-3-small"
        embeddings = AzureOpenAIEmbeddings(
            deployment=openai_api_deployment,
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            openai_api_version=os.environ['AZURE_OPENAI_VERSION'],
            openai_api_key=os.environ['AZURE_OPENAI_KEY'],
            # show_progress_bar=True,
            chunk_size = 1
        )
        return embeddings

    def _get_ai_search_index(self):
        ai_search_index = AzureSearch(
            azure_search_endpoint=self.azure_search_endpoint,
            azure_search_key=self.azure_search_key,
            index_name=self.azure_search_index_name,
            embedding_function=self.embedding_model.embed_query,
        )
        return ai_search_index

    def _get_azure_chat_model(self):
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

    def _load_api_vars(self):
        self.azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
        self.azure_openai_type = os.getenv('AZURE_OPENAI_TYPE')
        self.azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_completion_model = os.getenv('AZURE_OPENAI_COMPLETION_MODEL')
        self.azure_search_key = os.getenv('AZURE_AI_SEARCH_KEY')
        self.azure_search_index_name = os.getenv('AZURE_AI_SEARCH_INDEX_NAME')
        self.azure_search_service_name = os.getenv('AZURE_AI_SEARCH_SERVICE_NAME')
        self.azure_search_endpoint = \
            "https://" + self.azure_search_service_name + ".search.windows.net"

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
    inferencepipeline.infer(
        query=args.query,
        company_name=args.company_name,
        top_k=args.top_k
    )