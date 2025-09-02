from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace  
from dotenv import load_dotenv 
import os
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure the key is available
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")

llm  = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    huggingfacehub_api_token=huggingface_api_key
)


response = llm.invoke("what is Generative AI")

print(response)