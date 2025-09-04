from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()



model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

messages = [
    AIMessage(content="You are helpful AI Assistant"),
    HumanMessage(content="who is capital of pakistan")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)