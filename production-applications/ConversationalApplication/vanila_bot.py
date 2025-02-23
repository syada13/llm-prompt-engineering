
from langchain.schema import SystemMessage,AIMessage,HumanMessage
from langchain.chains import LLMChain,ConversationChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

from general_prompting_principles.recency_bias_order_matters import conversation

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

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm =chat,
    memory=memory,
    Verbose=True
)

# Conversation between Chat bot and user with ability to keep conversation history in memory
conversation.run("Hi there !")
conversation.run("What is the most iconic place in India?")
conversation.run("What kind of other events?")

#Make conversation interactive
while True:
    query = input('you:')
    if query == 'q':
        break
    output = conversation({'input':query})
    print('User', query)
    print('AI System', output['response'])