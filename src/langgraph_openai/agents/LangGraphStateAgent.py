from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

class LangGraphStateAgents:

    def invokeMessageStateAgent(llm: ChatOpenAI, state: MessagesState):
        respone = llm.invoke(state['messages'])
        return {'messages': [respone]}
