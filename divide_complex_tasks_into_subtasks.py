import os
import openai
from dotenv import load_dotenv
from prompting_with_clear_instructions import get_chat_completion

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message ="""
You are an AI assistant that summarizes articles.
To complete this task, do the following subtasks:
Read the provided article context comprehensively and identify the main topic and key points
Generate a paragraph summary of current article context that captures the essential information and conveys the main idea
Print each step of the process
Article:
"""

article="""
Recurrent neural networks, long short-term memory, and gated recurrent neural networks
in particular, […]
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": article}
]

# Call get_chat_completion method with prompt to get response from model
model_response = get_chat_completion(messages)
print(model_response)
