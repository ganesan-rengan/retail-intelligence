"""First LLM call. Prints the full response structure, not just the text."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["LLM_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="In one sentence, what is a retail stock keeping unit?",
)

print("=" * 60)
print("TEXT")
print("=" * 60)
print(response.text)

print()
print("=" * 60)
print("FULL RESPONSE OBJECT")
print("=" * 60)
print(response)

print()
print("=" * 60)
print("TOKEN USAGE")
print("=" * 60)
print(response.usage_metadata)