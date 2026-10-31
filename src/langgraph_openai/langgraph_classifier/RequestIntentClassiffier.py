from __future__ import annotations

from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class RequestIntentClassifier(BaseModel):
    human_prompt: Literal['chat','knowledge','code'] = Field(..., description="Classify the Intent of the human prompt")


class State(TypedDict):
    messages: Annotated[List, add_messages]
    human_prompt: str | None



def classify_intent(llm: str, state: State):    
    strucutred_llm = llm.with_structured_output(RequestIntentClassifier)
    result = strucutred_llm.invoke([
        {
            'role':'system',
            'content':'Determine|Classify is the request is to chat("CHAT"), knowledge("RAG") or code change("CODE")'},
        {'role':'user','content':state['messages'][-1].content}
    ])
    return {'human_prompt': result.human_prompt}


def prompt_llm_chat(llm: str,state: State):
    messages =  [{
                'role':'system',
                'content':'You are a talkative chat assistant. Always say you are a CHAT bot.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages':[{'role':'assistant','content':response.content}]}

def prompt_llm_knowledge(llm: str,state: State):
    messages =  [{
                'role':'system',
                'content':'You are a knowledge assistant. Always say you are a RAG bot.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages':[{'role':'assistant','content':response.content}]}

def prompt_llm_code(llm: str,state: State):
    messages =  [{
                'role':'system',
                'content':'You are a code assistant. Always say you are a CODE bot.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages':[{'role':'assistant','content':response.content}]}