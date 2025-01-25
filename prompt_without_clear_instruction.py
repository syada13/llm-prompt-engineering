import os
import openai
from dotenv import load_dotenv
from prompting_with_clear_instructions import get_chat_completion

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message = """
You are an AI assistant that helps humans by generating tutorials given a text.
You will be provided with text.If the text contains any kind of instructions on how to proceed with something, generate a tutorial in a bullet list.
Otherwise, inform the user that the text does not contain any instructions.
Text:
"""
vauge_instruction ="""
The sun is shining and dogs are running on the beach.
"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": vauge_instruction}
]

# Call get_chat_completion method with prompt to get response from model
model_response = get_chat_completion(messages)
print(model_response)

