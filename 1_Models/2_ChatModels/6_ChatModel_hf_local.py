from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs= dict(
        temprature = 0.5,
        max_new_token = 100
    )
)

model = ChatHuggingFace(llm = llm)

# response = model.invoke("what is generative ai?")

# print(response.content)