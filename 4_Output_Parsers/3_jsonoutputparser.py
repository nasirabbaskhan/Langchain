from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

# JsonOutputParser is used to get json formate response from LLM. but it does not follow the spacif schema.

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

parser = JsonOutputParser()

tamplate = PromptTemplate(
    template="Give me 5 facts about {topic}. \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = tamplate.format(topic="black hole")
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

chain = tamplate | model | parser
result = chain.invoke({"topic":"black hole"})
print(result)

