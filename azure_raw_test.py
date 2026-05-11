import requests
import json

# ==========================================
# 1. DIRECT AZURE CONFIGURATION
# ==========================================
API_KEY = "YOUR_API_KEY_HERE"

# THE FIX: We changed the URL from /chat/completions to /completions
ENDPOINT = "https://cropadvisoragent-resource.cognitiveservices.azure.com/openai/deployments/gpt-5.4-pro/completions?api-version=2024-12-01-preview"

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

# ==========================================
# 2. RAW PAYLOAD (Legacy Format)
# ==========================================
# We removed the 'messages' array and used the older 'prompt' format.
payload = {
    "prompt": "Hello, are you receiving this?",
    "max_tokens": 100
}

# ==========================================
# 3. EXECUTION
# ==========================================
print("Sending raw COMPLETIONS request directly to Azure API...")

try:
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    print(f"HTTP Status Code: {response.status_code}")
    
    print("\n--- Raw Azure Server Response ---")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Network Error: {e}")