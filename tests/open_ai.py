import openai
import os
import math
import json
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = """Extract the requester name from the text below:

Text: "Hi, this is Alice from FooCorp."
Return as JSON:
{
  "requester_name":
"""

response = client.completions.create(
    model="gpt-3.5-turbo-instruct", 
    prompt=prompt,
    max_tokens=10,
    temperature=0,
    logprobs=5,
)


text = response.choices[0].text.strip()
logprobs = response.choices[0].logprobs.token_logprobs
tokens = response.choices[0].logprobs.tokens

avg_logprob = sum(logprobs) / len(logprobs)
confidence = round(math.exp(avg_logprob), 3)

print("Extracted value:", text)
print("Tokens:", tokens)
print("Logprobs:", logprobs)
print("Confidence:", confidence)
