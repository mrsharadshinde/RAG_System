import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Initialize the Groq chat model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=SecretStr(api_key),
)

# Send a test message
response = llm.invoke([
    HumanMessage(content="Explain RAG in two simple sentences.")
])

print(response.content)
