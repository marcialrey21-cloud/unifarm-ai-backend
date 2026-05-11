import os
from openai import AzureOpenAI

# ==========================================
# 1. AZURE CONNECTION (Updated API Version)
# ==========================================
client = AzureOpenAI(
    api_version="2026-03-01", # <-- THE FIX: Updated to match your Resource JSON
    azure_endpoint="https://cropadvisoragent-resource.cognitiveservices.azure.com/",
    api_key="YOUR_API_KEY_HERE" # <-- Paste your key here!
)

print("Attempting connection with the 2026 API version...")

try:
    response = client.chat.completions.create(
        model="gpt-5.4-pro",
        messages=[
            {"role": "user", "content": "Hello! Please reply with a short greeting."}
        ]
    )
    
    print("\n--- Success! The server is accepting requests. ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nAzure Error Details: {e}")