from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# document loader are components in Langchain used to load data from various sources into a standerdized formate (usually Documents objects), which can then used for chunking, embedding, retriever and generation.

# TextLoader is a simple and commonly used document loader in LangChain that reads plain text (.txt) files and converts them into LangChain Documents objects.

# Use Case
# Ideal for loading chat logs, scraped text trascripts, code snippets or any plain text data into a langchain pipeline 

# works only with .txt files

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="summarize the given poem \n {poem}",
    input_variables=["poem"]
)

# load the text file
loader = TextLoader("cricket.txt", encoding='utf-8')

docs = loader.load()

# print("page_content",docs[0].page_content)
# print("metadata",docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"poem":docs[0].page_content})
print(result)