from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# RunnableSequence is a sequential chain of runables in langchain thaat execute each step one aftr another passing the output of one step as the input of the next

# It is useful when you need to compose multiple runables together in structured workflow


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template ="write a jok on the {topic}",
    input_variables=["topic"],
    
)

prompt2 = PromptTemplate(
    template="explain the following {jock}",
    input_variables=["kock"]
)
# chain = prompt | model | parser | prompt2 | model |parser

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic":"cricket"})

print(result)




