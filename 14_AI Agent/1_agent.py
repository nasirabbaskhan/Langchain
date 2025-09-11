from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

prompt = hub.pull("hwchase17/react")

search_tool = DuckDuckGoSearchRun()

# weather api tool
@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()

# agent
agent = create_react_agent(
  llm=llm,
  tools=[search_tool,get_weather_data],
  prompt= prompt
)

# agent executer 
agent_executor = AgentExecutor(
  agent=agent,
  tools=[search_tool,get_weather_data],
  verbose=True
)

response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)