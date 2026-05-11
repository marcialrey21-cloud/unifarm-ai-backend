import os
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. SECURE AZURE CONFIGURATION
# ==========================================
# We connect to your new, tool-friendly gpt-5.4 model.
API_KEY = "YOUR_API_KEY_HERE" # <-- Paste your key here!

llm = AzureChatOpenAI(
    azure_endpoint="https://cropadvisoragent-resource.cognitiveservices.azure.com/",
    azure_deployment="unifarm-agent-v1", # <-- MUST MATCH YOUR NEW DEPLOYMENT NAME
    openai_api_version="2024-12-01-preview", 
    api_key=API_KEY,
    temperature=0.1 # Allowed on standard models! Keeps answers factual.
)

# ==========================================
# 2. PRODUCTION TOOLS (Your DB Connectors)
# ==========================================
# These functions simulate querying your actual Uni-Farm Hub database.
@tool
def query_crop_database(crop_name: str) -> dict:
    """
    Query the Uni-Farm Hub database for current crop conditions and actions.
    Requires the name of the crop (e.g., 'tomato', 'rice').
    """
    print(f"\n[System] -> Executing SQL Query for Crop: {crop_name}...")
    if crop_name.lower() == "tomato":
        return {"status": "Vegetative", "soil_ph_reading": 6.5, "required_action": "Apply NPK 3-1-2"}
    elif crop_name.lower() == "rice":
        return {"status": "Sown Dec 21, 2025", "required_action": "Check Butachlor herbicide compatibility"}
    return {"status": "Unknown", "message": "Crop not found in database."}

@tool
def query_aquaculture_database(species_name: str) -> dict:
    """
    Query the Uni-Farm Hub database for aquaculture and pond metrics.
    Requires the species name (e.g., 'tilapia').
    """
    print(f"\n[System] -> Executing SQL Query for Pond: {species_name}...")
    if species_name.lower() == "tilapia":
        return {"water_temp_c": 28.0, "breeding_cycle": "Active", "required_action": "Identify male/female for fry management"}
    return {"status": "Unknown", "message": "Species not found in database."}

# Bundle the tools together for the AI to use
farm_tools = [query_crop_database, query_aquaculture_database]

# ==========================================
# 3. THE AGENT ENGINE
# ==========================================
# We initialize the LangGraph agent with the model and the tools.
unifarm_agent = create_react_agent(model=llm, tools=farm_tools)

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("=== Uni-Farm Hub AI Engine Online ===")
    print("Type 'exit' to quit.\n")
    
    # Standard models accept SystemMessages, which are perfect for defining the persona.
    system_instruction = SystemMessage(content=(
        "You are the Uni-Farm Hub AI Orchestrator. "
        "You manage a sustainable integrated farm in Cateel, Davao Oriental. "
        "You MUST use your tools to fetch real data before answering the user. "
        "Always synthesize the tool data into a clear, professional advisory report."
    ))
    
    while True:
        user_input = input("\nFarmer Query: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            print("Processing via Azure OpenAI...")
            
            # We pass both the system rules and the user's question to the agent
            user_message = HumanMessage(content=user_input)
            response = unifarm_agent.invoke({"messages": [system_instruction, user_message]})
            
            # Extract and print the AI's final answer after it uses the tools
            final_message = response["messages"][-1].content
            print("\n--- Uni-Farm Advisory ---")
            print(final_message)
            print("-" * 50)
            
        except Exception as e:
            print(f"\n[Error]: {e}")