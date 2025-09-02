from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()



model = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

response = model.invoke("what is machine learnig")

print(response.content)
