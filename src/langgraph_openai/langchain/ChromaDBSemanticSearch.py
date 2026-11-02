import os

from dotenv import load_dotenv
import chromadb

from langgraph_openai.domain.PydanticResearch import ResearchPydanticDomain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph_openai.agents.OpenAIStateLessAgent import OpenAIStateLessAgent
from langgraph_openai.prompts.OpenAIPromptTemplates import getResearchPydanticPrompt

load_dotenv()
#init_chat_model is the factory function, ChatOpenAI is its implementation
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

human_question = "What runs on water"

#in-memory client; embeddings are not persisted to disk
chroma_client = chromadb.EphemeralClient()
embeddings_collection = chroma_client.get_or_create_collection(name="embeddings_transport")
embeddings_collection.add(ids=['car','bus','plane','boat'],
                           documents=['car runs on land', 'bus runs on road', 'plane flies in the sky', 'boat sails on water'])

#results = embeddings_collection.query(query_texts=f'{human_question}',n_results=2, include=["embeddings"])
# include=["embeddings"] prints the embedding value
results = embeddings_collection.query(query_texts=f'{human_question}',n_results=2)
print(results)