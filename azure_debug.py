import os
from langchain_openai import AzureChatOpenAI

# ==========================================
# DIAGNOSTIC TEST: No Tools, Just Chat
# ==========================================
# We are testing if the model can respond to a basic greeting
# without the complex LangGraph tool orchestration.

try:
    print("Connecting to Azure OpenAI...")
    
    # Initialize the model with your exact dashboard details
    llm = AzureChatOpenAI(
        azure_endpoint="https://cropadvisoragent-resource.cognitiveservices.azure.com/",
        azure_deployment="gpt-5.4-pro",
        api_version="2026-03-05-preview", 
        api_key="YOUR_API_KEY_HERE", # Replace with your key
        # Note: We removed temperature=0, as some new models reject it.
    )

    # Send a simple text message to the model
    response = llm.invoke("Hello! Are you online and receiving messages?")
    
    print("\n--- Success! The Model Responded: ---")
    print(response.content)
    
except Exception as e:
    print(f"\nDiagnostic Error: {e}")