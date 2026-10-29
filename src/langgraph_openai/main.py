import os

from dotenv import load_dotenv

from langgraph_openai.domain.Pydantic_Domains import ResearchPydanticDomain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langgraph_openai.agents.ToolCallingAgent import ToolCallingAgent
from langgraph_openai.prompts.OpenAI_Prompt_Templates import getResearchPydanticPrompt

load_dotenv()

llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

prompt = getResearchPydanticPrompt(PydanticOutputParser(pydantic_object=ResearchPydanticDomain))

response = ToolCallingAgent.build_tool_calling_agent(llm, prompt,[])
print(response)