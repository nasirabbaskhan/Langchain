from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# RunnableLambda is runnable premitive that allows you to apply custom python functions with an AI pipeline.BrokenPipeError

# It acts as a middlwware between different AI components enabling processing, transformation, API calls, filtering, and post processing in angchain workflow.  


def word_count(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

parser = StrOutputParser()

promt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)


jock_gen_chain = RunnableSequence(promt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

chain = RunnableSequence(jock_gen_chain, parallel_chain)

result = chain.invoke({"topic":"AI"})

print("joke: ", result['joke'])
print("word_count: ", result['word_count'])