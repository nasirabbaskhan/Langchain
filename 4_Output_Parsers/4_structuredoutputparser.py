from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
load_dotenv()

# # StructuredOutputParser is an output parser in Langchain that helps extract structured JSON data from 
# LLM response base on the predefined field schemas.

# It works by defining a list of fields( ResponseSchema) that the model should return, ensuring the 
# output follows  a structured formate.

## But it does not do data validation


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

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
    ResponseSchema(name="fact_4", description="Fact 4 about the topic"),
    ResponseSchema(name="fact_5", description="Fact 5 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Give me 5 facts about {topic}. \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()},
    validate_template=True
)

# prompt = template.format(topic="black hole")
# response = model.invoke(prompt)
# final_response = parser.parse(response.content)
# print(final_response)

chain = template | model | parser
response = chain.invoke({"topic": "black hole"})
print(response)