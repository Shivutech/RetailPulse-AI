import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found")
    raise SystemExit(1)

client = genai.Client(api_key=api_key)

models_to_try = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash"
]

for model_name in models_to_try:

    print(f"\nTrying model: {model_name}")

    try:

        response = client.models.generate_content(
            model=model_name,
            contents="Explain a retail dashboard in one short sentence."
        )

        print("\nSUCCESS!")
        print("Model:", model_name)
        print("Response:", response.text)

        break

    except Exception as e:

        print("Model unavailable:", model_name)
        print("Error:", str(e)[:250])