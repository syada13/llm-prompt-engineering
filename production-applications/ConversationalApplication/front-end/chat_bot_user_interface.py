#1. Setting the configuration of the webpage:
import streamlit  as st
from streamlit.external.langchain import StreamlitCallbackHandler

from general_prompting_principles.generate_multiple_answers_and_choose_the_best import response

st.set_page_config(page_title="Travlet")
st.header(('Welcome to Travlet, your travel assistant with Internet access. What are you planning for your next trip?'))

#2 Initializing the LangChain backbone components we need.
from langchain import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents.agent_toolkits import create_retriever_tool,create_conversational_retrieval_agent
import os
from dotenv import load_dotenv

load_dotenv()
serpapi_api_key =os.getenv("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key)
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

llm = ChatOpenAI()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(),
        "italy_travel",
        "Searches and returns documents regarding Italy."
    )]

agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)


#3 -Setting the input box for the user with a placeholder question:
user_query = st.text_input("**Where are you planning your next vacation?**", placeholder="Ask me anything!")

#4 - Setting Streamlit’s session states. Session state is a way to share variables between reruns, for each user session.
# We can use the session state API to initialize, read, update, and delete variables in the session state

if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "memory" not in st.session_state:
    st.session_state["memory"] = memory

#5 - Display the whole conversation.
# For each message, it creates a Streamlit element called st.chat_message that displays a chat message in a nice format:
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

#6 -Configuring the AI assistant to respond when given a user’s query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response= agent(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


#7 - Add a button to clear the history of the conversation and start from scratch:
if st.sidebar.button('"Reset chat history"'):
    st.session_state.messages = []






