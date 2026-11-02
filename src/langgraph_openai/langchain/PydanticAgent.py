import os

from dotenv import load_dotenv

from langgraph_openai.domain.PydanticResearch import ResearchPydanticDomain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph_openai.agents.OpenAIStateLessAgent import OpenAIStateLessAgent
from langgraph_openai.prompts.OpenAIPromptTemplates import getResearchPydanticPrompt

load_dotenv()
#init_chat_model is the factory function, ChatOpenAI is its implementation
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

human_question = "What is the capital of India?"


prompt = getResearchPydanticPrompt(PydanticOutputParser(pydantic_object=ResearchPydanticDomain), human_question)

response = OpenAIStateLessAgent.invokeStatelessAgent(llm, prompt, [])
print(response)