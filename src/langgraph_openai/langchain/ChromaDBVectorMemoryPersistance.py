import os

from dotenv import load_dotenv
import chromadb

from langgraph_openai.domain.PydanticResearch import ResearchPydanticDomain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph_openai.agents.OpenAIStateLessAgent import OpenAIStateLessAgent
from langgraph_openai.prompts.OpenAIPromptTemplates import getResearchPydanticPrompt, simpleChatPrompt

load_dotenv()
#init_chat_model is the factory function, ChatOpenAI is its implementation
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"))

human_question = "What is the capital of India?"

#in-memory client; embeddings are not persisted to disk
chroma_client = chromadb.EphemeralClient()
embeddings_collection = chroma_client.get_or_create_collection(name="embeddings_project")
embeddings_collection.add(ids=[human_question], documents=[human_question])

stored = embeddings_collection.get(ids=[human_question], include=["embeddings"])
vector = stored["embeddings"][0]
print(f"Vector dimensions: {len(vector)}")
print(f"Vector preview (first 10 dims): {vector[:10]}")

prompt = getResearchPydanticPrompt(PydanticOutputParser(pydantic_object=ResearchPydanticDomain), human_question)

response = OpenAIStateLessAgent.invokeStatelessAgent(llm, prompt, [])
print(response)

# ---------------------------------------------------------------------------
# Real-time example: agent memory persistence (Option C - summarize a whole
# session into dense facts first, then embed only the summary).
# Embedding all 10 raw turns as one blob would blur the vector (small talk +
# facts mixed together); embedding each turn separately is 10x storage/search
# cost. Summarizing keeps 1 embedding per session while staying search-friendly.
# ---------------------------------------------------------------------------
memory_collection = chroma_client.get_or_create_collection(name="agent_memory")

# Simulated 10-turn back-and-forth from a single session.
conversation_turns = [
    "User: Can you help me plan meals for the week?",
    "Agent: Sure, what are your dietary preferences?",
    "User: I'm vegetarian and I'm allergic to nuts.",
    "Agent: Got it, I'll avoid nuts and meat in suggestions.",
    "User: Also I prefer low-spice food.",
    "Agent: Noted, I'll keep spice levels mild.",
    "User: What's a good breakfast option?",
    "Agent: How about oatmeal with fruit?",
    "User: Sounds good, thanks!",
    "Agent: You're welcome!",
]
full_transcript = "\n".join(conversation_turns)

# Ask the LLM to compress the transcript into the lasting facts worth remembering.
summarize_prompt = simpleChatPrompt(
    "Summarize only the lasting facts/preferences about the user from this "
    f"conversation in 1-2 short sentences, ignore small talk:\n{full_transcript}"
)
summary_response = OpenAIStateLessAgent.invokeStatelessAgent(llm, summarize_prompt, [])
session_summary = summary_response["messages"][-1].content  # dense facts distilled from all 10 turns

# One embedding represents the whole session instead of 10 separate ones.
memory_collection.add(
    ids=["user123_summary_2026-11-17"],
    documents=[session_summary],
    metadatas=[{"user_id": "123", "type": "session_summary", "source_session": "2026-11-17-session-1"}],
)
print(f"Stored session summary: {session_summary}")

# get memory by metadata for a specific user
user_memories = memory_collection.get(where={"user_id": "123"}, include=["documents", "metadatas"])
print(f"All memories for user 123: {user_memories['documents']}")

# Session 2 (e.g. Friday, brand-new conversation): the new question never
# mentions "nuts" or "allergy", but the agent still needs to recall the fact
# that was buried inside the 10-turn transcript from Session 1.
new_user_query = "Can you suggest a snack for me?"

# where= scopes the search to only this user's memories (not everyone else's)
recalled = memory_collection.query(
    query_texts=[new_user_query],
    n_results=1,
    where={"user_id": "123"},
)
recalled_memory_text = recalled["documents"][0][0]
recalled_distance = recalled["distances"][0][0]  # lower = more semantically similar

print(f"User asked: {new_user_query}")
print(f"Recalled memory (matched by meaning, not exact words): {recalled_memory_text}")
print(f"Similarity distance: {recalled_distance:.4f}")

# Feed the recalled memory back into the prompt as extra context so the LLM's
# answer respects it even though this turn's question never mentioned it.
memory_aware_question = (
    f"User question: {new_user_query}\n"
    f"Known fact about this user (from memory): {recalled_memory_text}\n"
    "Answer the question while respecting the known fact."
)
memory_prompt = simpleChatPrompt(memory_aware_question)
memory_response = OpenAIStateLessAgent.invokeStatelessAgent(llm, memory_prompt, [])
print(memory_response)