from langchain_huggingface import  HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Deli is the captal of India"

vector = embedding.embed_query(text)

print(len(vector))       # show vector size
print(vector[:10])    