from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


load_dotenv()


class Weather(BaseModel):
    temp: float = Field(..., description="Current temperature")
    feels_like: float = Field(..., description="Apparent temperature")
    humidity: int = Field(..., description="Humidity percentage", ge=0, le=100)
    description: str = Field(..., description="Summary (e.g., 'Partly Cloudy')")
    wind_speed: float = Field(..., description="Wind speed in km/h")

llm = ChatGroq(model="llama-3.3-70b-versatile")
tools=[TavilySearch()]



agent=create_agent(model=llm,tools=tools,response_format=Weather)

message = [SystemMessage(content="You are a professional Senior Meteorologist. Your goal is to provide highly accurate weather forecasts and atmospheric analyses based on provided data or locations."),
           HumanMessage(content="What is the weather of Tokyo?")
           ]

response=agent.invoke({"messages": message})


print(response)