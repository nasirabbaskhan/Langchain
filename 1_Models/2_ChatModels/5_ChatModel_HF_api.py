from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure the key is available
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",
                         task="text-generation",
                         huggingfacehub_api_token=huggingface_api_key
                          )

chat_model = ChatHuggingFace(llm = llm)

response = chat_model.invoke("what is generative ai?")

print(response.content)