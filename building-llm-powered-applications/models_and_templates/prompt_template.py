from langchain.llms import OpenAI
from langchain import PromptTemplate

llm = OpenAI(openai_api_key)
print(llm("tell me a joke"))

template ="""Sentence:{sentence}
Translation in {language}:"""

prompt= PromptTemplate(template=template, input_variables=["sentence","language"])
print(prompt.format(sentence=" the cat is on the table", language ="spanish"))

