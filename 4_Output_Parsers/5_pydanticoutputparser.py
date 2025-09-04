from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import Annotated
from dotenv import load_dotenv
import os
load_dotenv()

# PydanticOutputParser is structured output parser in Langchain that Pydantic model to enforce scema validation 
# when processing LLM response

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

class Person(BaseModel):
    name: Annotated[str, Field(..., description="Name of the person'")]
    age:Annotated[int, Field(..., description="Age of the person")]
    city:Annotated[str, Field(..., description='Name of the city the person belongs to')]


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}',",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
    validate_template=True
)

chain = template | model | parser

response = chain.invoke({"place":"sri lanka"})
print(response)