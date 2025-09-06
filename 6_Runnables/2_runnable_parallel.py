from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# RunnableParallel is a runable premitive that allows multiple runnables to execute parallel.

# Each runnable receive the same input and processes independantly, producing a dictionary  of outputs.


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


pareser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed 400 words tweet about {topic}",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="generae the Linkedin post on {topic}",
    input_variables=["topic"]
)



parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, pareser),
    'linkedin': RunnableSequence(prompt2, model, pareser)
})

result = parallel_chain.invoke({"topic":"cricket"}) # it gives 2 result of dictionary because it has 2 chains 

print(result["tweet"])
print("_______________________")
print(result["linkedin"])