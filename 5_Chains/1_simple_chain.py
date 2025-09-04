from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

template = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
    validate_template=True
)


parserr = StrOutputParser()

chain = template | model | parserr

respone = chain.invoke({"topic":"Cricket"})

print(respone)

# to viulize the graph
chain.get_graph().print_ascii()