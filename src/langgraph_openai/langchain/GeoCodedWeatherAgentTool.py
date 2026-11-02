import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
from langgraph_openai.agents.OpenAIStateLessAgent import OpenAIStateLessAgent


load_dotenv()
#init_chat_model is the factory function, ChatOpenAI is its implementation
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_4.1"), temperature=float(os.getenv("TEMPERATURE")))

city = 'London'


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful weather assistant. Use the available tools to answer weather questions."),
    ("user", f"What is the weather like in {city}?"),
])

#call the geocded api; get the latitude and longitude, then fetch the weather using those coordinates
# relay the weather information to the model 
@tool('get_weather_by_city', description="Get the weather by input city name")
def get_weather_by_city(city:str) -> str:
    try:
        geo_response = requests.get(f'https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1', timeout=10)
        geo_response.raise_for_status()
        results = geo_response.json().get("results")
        if not results:
            return f"Could not find location data for {city}"

        latitude = results[0]["latitude"]
        longitude = results[0]["longitude"]

        weather_response = requests.get(
            'https://api.open-meteo.com/v1/forecast',
            params={"latitude": latitude, "longitude": longitude, "current_weather": True},
            timeout=10,
        )
        weather_response.raise_for_status()
        return weather_response.json()
    except requests.exceptions.RequestException as e:
        return f"Unable to fetch weather for {city}: {e}"


raw_response = OpenAIStateLessAgent.invokeStatelessAgent(llm, prompt, [get_weather_by_city])
print(raw_response)
