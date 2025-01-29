import os
import openai
from dotenv import load_dotenv
from prompting_with_clear_instructions import get_chat_completion

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

system_message ="""
You are a Python expert who produces python code based on user's request.
===>START EXAMPLE
---User Query---
Give me a function to print a text.
---User Output---
Below you can find the described function:
```def my_print(text):
     return print(text)
```
<===END EXAMPLE
"""

query ="generate a Python function to calculate the nth Fibonacci number"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": query}
]


# Call get_chat_completion method with prompt to get response from model
model_response = get_chat_completion(messages)
print(model_response)