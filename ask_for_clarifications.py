import os
import openai
from dotenv import load_dotenv
from prompting_with_clear_instructions import get_chat_completion

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message ="""
You are an AI assistant specialized in solving riddles.
Given a riddle, solve it the best you can.
Provide a clear justification of your answer and the reasoning behind behind it.
Riddle:
"""

riddle="""
What has a face and two hands, but no arms or legs?
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": riddle}
]


model_response = get_chat_completion(messages)
print(model_response)
