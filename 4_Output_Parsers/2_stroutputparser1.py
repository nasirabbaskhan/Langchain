from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()


# StrOutputParser is simpist output parser in langchain. It is used to parse the output of the Language model (LLM) and return it as a plain string. it return directly only content string
huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ensure the key is available
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingface_api_key
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# detailed reprt
template_1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
    validate_template=True
)

# summary
template_2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
    validate_template=True
)


chain = template_1 | model | parser | template_2 | model | parser

response = chain.invoke({"topic":"black hole"})

print(response) # because we use stroutputparser so we do not have need to use response.content
