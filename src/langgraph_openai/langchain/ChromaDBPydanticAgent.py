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

human_question = "What is the capital of India?"

#in-memory client; embeddings are not persisted to disk
chroma_client = chromadb.EphemeralClient()
embeddings_collection = chroma_client.get_or_create_collection(name="embeddings_project")
embeddings_collection.add(ids=[human_question], documents=[human_question])

stored = embeddings_collection.get(ids=[human_question], include=["embeddings"])
vector = stored["embeddings"][0]
print(f"Vector dimensions: {len(vector)}")
print(f"Vector preview (first 10 dims): {vector[:10]}")

prompt = getResearchPydanticPrompt(PydanticOutputParser(pydantic_object=ResearchPydanticDomain), human_question)

response = OpenAIStateLessAgent.invokeStatelessAgent(llm, prompt, [])
print(response)