from langchain_core.prompts import ChatPromptTemplate



# ChatPromptTemplate is used to create dynamic prompt for multiples input


chat_template = ChatPromptTemplate({
    ("system", "You are a helpful {ai_domain} expert"),
    ('human', 'Explain in simple terms, what is {topic}')
})


prompt = chat_template.invoke({"ai_domain":"cricket", "topic":"dusra"}) # multiple inputes

print(prompt)