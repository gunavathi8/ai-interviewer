import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

# Test the query
response = llm.invoke("Generate a 3 simple Python interview questions.")
print(response.content)