from typing import List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ToolCallingAgent:

    def build_tool_calling_agent(llm:ChatOpenAI, prompts:ChatPromptTemplate, tools:List) -> str:
        agent = create_agent(llm, tools=tools, system_prompt=prompts.messages[0].prompt.template)

        raw_response = agent.invoke({"messages": [{"role": "user", "content": "What is the capital of Japan?"}]})
        return raw_response
