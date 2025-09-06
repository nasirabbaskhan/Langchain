from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# RunnablePassthrough is a special Runnable premitive that simply returns the input as output without modifying (changing or processing) It is use as placeholder 


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="generate the jock on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate the explination on the {jock}",
    input_variables=["jock"]
)

jock_gen_chain = RunnableSequence(prompt1, model, parser)

parralel_chain = RunnableParallel({
    'jock': RunnablePassthrough(),
    "explination": RunnableSequence(prompt2,model,parser)
})


chain = RunnableSequence(jock_gen_chain, parralel_chain)
result = chain.invoke({"topic":"AI"})

print("jock: ", result["jock"])
print("explination: ", result["explination"])


