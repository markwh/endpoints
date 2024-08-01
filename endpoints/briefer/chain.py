from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WeatherDataLoader
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.base import RunnableLambda

OPENWEATHERMAP_API_KEY = os.environ['OPENWEATHERMAP_API_KEY']


def retrieve_weather(input_dict):
  location = input_dict['location']
  loader = WeatherDataLoader.from_params([location], openweathermap_api_key=OPENWEATHERMAP_API_KEY)
  documents = loader.load()
  return documents[0].page_content

prompt_template = PromptTemplate.from_template("""

Write a daily briefing for the user. Include a motivational greeting, followed by a summary of the following information:
  
WEATHER: {weather_summary}""".strip()
)

gpt_4o = init_chat_model("gpt-4o", model_provider="openai")
llm = gpt_4o

chain = retrieve_weather | prompt_template | llm | StrOutputParser()
