from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# message_placeholder is used to store and retrieve previous chat history so that llm give response from previous chat


chat_prompt = ChatPromptTemplate([
    ("system", "you are a helpful AI Assistant"),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user","{query}")
    ])

chat_history = []

with open("chat_history.txt") as f:
    chat_history.append(f.read())

prompt =  chat_prompt.invoke({"chat_history":chat_history, "query":"Where is my refund" })

print(prompt)


