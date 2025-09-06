from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader is a document loader in LangChain used to load content from PDF files and convert each page into a document object.

# [
#     Document(page_content="text from page 1", metadata={"page":0, "sourse":"file.pde"})
#     Document(page_content="text from page 2", metadata={"page":1, "sourse":"file.pde"})
# ]

# It use PyPDF library under the hood - not great with scanned PDF's or complex layouts'


loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

print("page_content",docs[0].page_content) # fist page from 23 page documents
print("metadata",docs[0].metadata)

print("Length",len(docs))