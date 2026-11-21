import os
import time

import numpy as np
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

cached_embeddings = []
cached_responses = []


questions = [
    "What is the capital of France?",
     "What is the capital of India?",
     "What is the capital of France?",
     "Tell me france country capital?"
]
for question in questions:
    start = time.perf_counter()
    embedding = client.embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL_SMALL"),
        input=question
    ).data[0].embedding
    similarity = 0
    if cached_embeddings:
        #@ for matrix multiplication, creating numpy array for efficency
        similarities = np.array(cached_embeddings) @ embedding
        bestMatch = np.argmax(similarities)

        similarity = similarities[bestMatch]

    if similarity > 0.9:
        print(f"Similarity : {similarity} & Value from cache :  {cached_responses[bestMatch]}")
    else:
        response = client.responses.create(
        model=llm.model_name,
        instructions= 'Answer in one sentence.',
        input=question
        ).output_text
        print(f"Similarity : {similarity} & Value from Model: {response}")
        cached_embeddings.append(embedding)
        cached_responses.append(response)

    print(time.perf_counter() - start)

# LOCAL RUN INSTRUCTIONS ::  python3  SemanticCaching_TextEmbeddingCaching.py   
# LOCAL RUN INSTRUCTIONS ::  path + uv run  SemanticCaching_TextEmbeddingCaching.py   


    
