from dotenv import load_dotenv
import os
# Read .env file and load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

