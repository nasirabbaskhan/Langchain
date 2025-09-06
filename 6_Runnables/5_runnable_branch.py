from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# RunnableBranch is a control flow    component in Langchain that allows you to conditionally route input data to different chains or runnables based on custom logic.

# It functions like an if/elif/else block for chainn - where you define a set of condition functions, each associated with a runnable (e.g. LLM calls, prompt, chain or tool). The first matching condition is executed. if no condition matches, a default runnable is used (if provided).


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)
report_chain = RunnableSequence(prompt1 , model, parser)

branch_chain = RunnableBranch(
(lambda x: len(x.split())>300 , RunnableSequence(prompt2, model, parser)),
RunnablePassthrough()
)

chain = RunnableSequence(report_chain, branch_chain)

result = chain.invoke({"topic":"AI"})

print(result)

