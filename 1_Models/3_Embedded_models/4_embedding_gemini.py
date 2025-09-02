from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding =  GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

result = embedding.embed_query("langchain is framework")

print(str(result))