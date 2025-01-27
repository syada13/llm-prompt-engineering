import os
import openai
from dotenv import load_dotenv
from prompting_with_clear_instructions import get_chat_completion

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message = """
You are an AI assistant specialised in solving riddles.
Given a riddle,you have to generate three answers to the riddle.
For each answer, be specific about your reasoning you made.
Then, among the three answers, select the one that is plausible given the riddle.

Riddel:
"""

riddle ="""
What has a face and two hands, but no arms and legs?
"""

messages = [
    {"role":"system", "content": system_message},
    {"role":"user", "content": riddle}
           ]


response = get_chat_completion(messages)
print(response)