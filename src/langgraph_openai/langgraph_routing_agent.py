import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.graph import MessageState, StateGraph, START, END


load_dotenv()

llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))