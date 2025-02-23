
from langchain.schema import SystemMessage,AIMessage,HumanMessage
from langchain.chains import LLMChain,ConversationChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
temperature = 0

chat = ChatOpenAI(temperature = 0)
messages = [
    SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
    HumanMessage(content="I'm going to India for 2 days, what can I visit?")
]

output = chat(messages)
print(output.content)