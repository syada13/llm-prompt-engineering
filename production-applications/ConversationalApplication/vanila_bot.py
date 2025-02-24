
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


# Add a non-parametric knowledge
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load Italy_travel.pdf
raw_documents = PyPDFLoader("italy_travel.pdf").load()

#Initialize TextSplitter to split loaded document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

# Split raw_documents using Text Splitter
documents = text_splitter.split_documents(raw_documents)

#Embed text chunks using the by default embedding model = text-embedding-ada-002 and save into FAISS vector store.
db = FAISS.from_documents(documents,OpenAIEmbeddings())

#Use ConversationBufferMemory to keep chat history in memory
memory = ConversationBufferMemory(
    memory_key ='chat_history',
    return_messages=True
)

# Let's interact with Conversation now.
# The ConversationalRetrievalChain uses a prompt template called CONDENSE_QUESTION_PROMPT,
# which merges the last user’s query with the chat history, so that it results as just one query to the retriever.
# If you want to pass a custom prompt, you can do so using the condense_question_prompt parameter in the ConversationalRetrievalChain.from_llm module.

query_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=db.as_retriever(),
    memory=memory,
    verbose=True
)

query_chain.run({'question':'Give me some review about the Tajmahal'})
