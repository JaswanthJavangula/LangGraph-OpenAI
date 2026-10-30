import os
import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph_openai.agents.LangGraphStateAgents import LangGraphStateAgents
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
# init_chat_model is the factory function to initialize the chat model with the specified OpenAI model.
llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))

human_question = "What is the capital of India?"

#uses in-mem for checkpoint saving and context management
checkpoint_saver = InMemorySaver()
#same thread_id will be used for all invocations within this session to retain context
config = {'configurable':{'thread_id':uuid.uuid4()}}

def prompt_llm(state: MessagesState):
    respone = LangGraphStateAgents.invokeMessageStateAgent(llm, state)
    return respone


graph_builder = StateGraph(MessagesState)
graph_builder.add_node('prompt_llm', prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph_instance = graph_builder.compile(checkpointer = checkpoint_saver)


print(graph_instance.invoke({'messages': [{'role': 'user', 'content': human_question}]} , config=config))