from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template='Generate the detailed report on {topic}',
    input_variables=["topic"],
    validate_template=True
)

prompt_2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text: \n {text}',
    input_variables=["text"],
    validate_template=True
)

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({"topic":"unemployment in pakistan"})

print(result)

chain.get_graph().print_ascii()

