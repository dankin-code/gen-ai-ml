from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

# OpenAI API Models Available
#   Resource: https://platform.openai.com/docs/models/model-endpoint-compatibility
response = client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a helpful language translating assistant."},
        {"role": "user", "content": "Translate the following English text to French: 'The recent Nike earnings call was upbeat'"},
    ],
    max_tokens=60
)

print(response.choices[0].message.content)