import os
from dotenv import load_dotenv
import openai

# Read .env file and load environment variables
load_dotenv()

# Read OPEN_API_KEY and configure to be sent via API call to model
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create messages as required by the API
system_message = """
You are an AI assistant that helps humans by generating tutorials given a text.
You will be provided with text.If the text contains any kind of instructions on how to proceed with something, generate a tutorial in a bullet list.
Otherwise, inform the user that the text does not contain any instructions.
Text:
"""
instructions_1 = """
To prepare the known sauce from Genova, Italy, you can start by toasting the pine nuts to then coarsely
chop them in a kitchen mortar together with basil and garlic. Then, add half of the oil in the kitchen mortar and season with salt and pepper.
Finally, transfer the pesto to a bowl and stir in the grated Parmesan cheese.
"""

#The chat model comes with two variables placeholders:
# system message: how we want our model to behave(in this case an AI Assistant)
# and instructions (or query), where the user will ask the model its questions.
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": instructions_1}
]


def get_chat_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model= model,
        messages= messages,
        temperature= 0
    )
    return response.choices[0].message['content']


# Call get_chat_completion method with prompt to get response from model
model_response = get_chat_completion(messages)
print(model_response)

