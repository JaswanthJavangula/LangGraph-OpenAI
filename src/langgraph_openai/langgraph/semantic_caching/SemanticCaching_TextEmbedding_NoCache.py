import os
import time

from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))
questions = [
    "What is the capital of France?",
     "What is the capital of France?"
]
for question in questions:
    start = time.perf_counter()
    response = client.responses.create(
        model=llm.model_name,
        instructions= 'Answer in one sentence.',
        input=question
    ).output_text

    print(time.perf_counter() - start)
    print(response)

# LOCAL RUN INSTRUCTIONS ::  path + uv run  SemanticCaching_TextEmbedding_NoCache.py   


    
