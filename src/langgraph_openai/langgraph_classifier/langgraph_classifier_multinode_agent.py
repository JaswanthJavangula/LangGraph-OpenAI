import os
import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import  StateGraph, START, END
from langgraph_openai.langgraph_classifier.RequestIntentClassiffier import (
    State,
    classify_intent,
    prompt_llm_chat,
    prompt_llm_knowledge,
    prompt_llm_code,
)
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
# init_chat_model is the factory function to initialize the chat model with the specified OpenAI model.
llm = init_chat_model(model=os.getenv("OPENAI_MODEL"))


#uses in-mem for checkpoint saving and context management
checkpoint_saver = InMemorySaver()
#same thread_id will be used for all invocations within this session to retain context
config = {'configurable':{'thread_id':uuid.uuid4()}}


graph_builder = StateGraph(State)
graph_builder.add_node('classifier_agent', lambda state: classify_intent(llm, state))
graph_builder.add_node('chat_agent', lambda state: prompt_llm_chat(llm, state))
graph_builder.add_node('knowledge_agent', lambda state: prompt_llm_knowledge(llm, state))
graph_builder.add_node('code_agent', lambda state: prompt_llm_code(llm, state))

#edges
graph_builder.add_edge(START, 'classifier_agent')

def route_by_intent(state: State) -> str:
    return state['human_prompt']

graph_builder.add_conditional_edges('classifier_agent', route_by_intent, {
    'chat': 'chat_agent',
    'knowledge': 'knowledge_agent',
    'code': 'code_agent',
})
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('knowledge_agent', END)
graph_builder.add_edge('code_agent', END)

graph_instance = graph_builder.compile(checkpointer = checkpoint_saver)

while True:
    user_input = input("Enter your question: ")
    result = graph_instance.invoke({'messages': [{'role':'user', 'content': user_input}]}, config=config)
    print(result['messages'][-1].content)