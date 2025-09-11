from langchain_community.llms import ollama
from dotenv import load_dotenv

# ollama helps you to using open source models

# first you have need to download ollama and then download open source model locally through ollama

# command:  `ollama run llama2`

load_dotenv()

llm = ollama(model="llama2")

response = llm.invoke("what is langchain")

print(response)