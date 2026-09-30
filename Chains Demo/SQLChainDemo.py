# pip install langchain-openai

from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

#Load the environment variables from .env files
load_dotenv()

# Read the key
api_key = os.getenv('API_KEY')

os.environment["OPEN_API_KEY"] = api_key

#Initialize LLM
llm = ChatOpenAI()

db = SQLDatabase.from_uri("sqlite://Chinook.db")

chain = create_sql_query_chain(llm, db)

response = chain.invoke({"question": "How many employees are there"})

print(response)


## This is a paid model and we must buy subcription to run the code
# You can revise the code with free version like Huggigface likely to make it work 