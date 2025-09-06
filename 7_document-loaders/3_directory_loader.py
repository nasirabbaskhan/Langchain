from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# PyPDFLoader is a document loader that lets you load multiple documents from a dictionary (folder or files)
loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)


docs = loader.load()  # usin for small documents
# docs = loader.lazy_load() # use for large documents 0r streaming 

print(docs[0].page_content)
print(docs[0].metadata)

for documents in docs:
    print(documents.metadata)

print(len(docs))