from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# PromptTemplate is used to create dynamic prompt for single input

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

template = PromptTemplate(
    template="Greet this person in 5 languages, The name of person is {name}",
    input_variables=["name"],
    validate_template=True
)


# prompt = template.invoke({"name":"anasir"})

# result = model.invoke(prompt)

chain = template | model
result = chain.invoke({"name":"anasir"})

print(result.content)