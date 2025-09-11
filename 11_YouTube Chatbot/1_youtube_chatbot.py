# llm and embedding model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# RAG
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# prompt 
from langchain.prompts import PromptTemplate
# cahin
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# OutputParser
from langchain_core.output_parsers import StrOutputParser
# load the env variables
from dotenv import load_dotenv

# 1:indexing -> 2:RAG
# indexing -> Retrieval -> Augmentation -> Generation

# 1: indexing (1:document loader, 2:text splitting, 3:embedding, 4:vector store)

# 2: RAG (Retrieval Augmented Generation)
# Retrieval: used to retrieve the most relevent documents fron vactor store base on the user's query
# Augmentation: process to create the prompt that have user query (question) and most reevent documents (context) that Retrieval has retrive
# Generation: process of generate the response from llm by using prompt that having question and context


load_dotenv()

# llm
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# parser
parser = StrOutputParser()


# indexing
# transcript  fetching
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.fetch(video_id=video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# embedding
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

# 2 RAG
# Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}) 

# Building a Chain
# In chain we automate the process of: Retrieval -> Augmentation -> Generation

# Augmentation: prompt with user's question and retriever's response from vectorstores and both will pass to llm to get final response
prompt = PromptTemplate(
   template= """ 
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}

""",
input_variables=["context", "question"]

)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parraleel_cahin = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs), # this chain retrive te context documents and then return the text
    "question": RunnablePassthrough(),

})

 # parraleel_cahin: Retrieval -> prompt: Augmentation -> llm: Generation
chain = parraleel_cahin | prompt | llm | parser 


# Generation: generate the response from llm by using prompt that having question and context
result = chain.invoke('Can you summarize the video')

print(result)

