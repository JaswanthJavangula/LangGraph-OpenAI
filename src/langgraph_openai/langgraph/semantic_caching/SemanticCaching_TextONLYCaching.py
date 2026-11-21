import hashlib
import os
import time

from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

cache = {}

questions = [
    "What is the capital of France?",
     "What is the capital of France?"
]

for question in questions:
    start = time.perf_counter()
    #cache-key
    sha_hashed_question_test = hashlib.sha256(question.encode()).hexdigest()

    if sha_hashed_question_test in cache:
        print(f" value from cache : {cache[sha_hashed_question_test]}")
    else:
        response = client.responses.create(
            model=llm.model_name,
            instructions= 'Answer in one sentence.',
            input=question
        ).output_text
        cache[sha_hashed_question_test] = response
        print(f"Value from Model : {response}")
        print(time.perf_counter() - start)

# LOCAL RUN INSTRUCTIONS ::  python3  SemanticCaching_TextONLYCaching.py   
# LOCAL RUN INSTRUCTIONS ::  path + uv run  SemanticCaching_TextONLYCaching.py   


    
