from langchain_community.document_loaders import CSVLoader

# CSVLoader is a document loader used to load CSV files into LangChain Document objects - one per row , by default 


loader = CSVLoader("Social_Network_Ads.csv")


docs = loader.load()

print(docs)

print(docs[3].page_content)

print(len(docs))


