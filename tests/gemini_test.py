import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API
genai.configure(api_key=GEMINI_API_KEY)

try:
    # List available models
    models = genai.list_models()
    print("✅ Available Models:")
    for m in models:
        print(f" - {m.name}")

    # Use a valid model for a simple prompt
    model = genai.GenerativeModel("models/gemini-1.5-pro")

    response = model.generate_content("Say hello to the world in a cool way.")

    print("\n✅ Gemini Response:")
    if hasattr(response, "text"):
        print(response.text)
    else:
        print(response)

except Exception as e:
    print("\n❌ Gemini API call failed:")
    print(e)
