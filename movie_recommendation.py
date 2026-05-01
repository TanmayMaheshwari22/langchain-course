from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class MovieRecommendation(BaseModel):
    title: str = Field(description="The name of the movie")
    year: int = Field(description="The release year")
    genres: List[str] = Field(description="List of genres (e.g., Sci-Fi, Drama)")
    rating: float = Field(description="The IMDb or Rotten Tomatoes rating out of 10")
    reasoning: str = Field(description="A brief explanation of why this movie fits the request")


llm = ChatGroq(model="llama-3.3-70b-versatile")

structured_llm = llm.with_structured_output(MovieRecommendation)

messages = [
    SystemMessage(content="You are an expert cinema critic. Only recommend highly-rated, critically acclaimed movies."),
    HumanMessage(content="Recommend all time best movie")
]

# 4. Invoke with the list of messages
recommendation = structured_llm.invoke(messages)


# # Accessing the data cleanly
print(f"Movie: {recommendation.title} ({recommendation.year})")
print(f"Rating: {recommendation.rating}/10")
print(f"genres: {recommendation.genres}")
print(f"Why you'll like it: {recommendation.reasoning}")

