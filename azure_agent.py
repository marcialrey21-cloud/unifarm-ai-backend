import os
import json
from openai import AzureOpenAI

# ==========================================
# Step 1. AI MODEL SETUP 
# ==========================================
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://cropadvisoragent-resource.cognitiveservices.azure.com/",
    api_key="YOUR_API_KEY_HERE" # <-- Paste your key here!
)

deployment_name = "gpt-5.4-pro"

# ==========================================
# Step 2. DEFINING OUR PYTHON TOOLS
# ==========================================
def get_soil_data(location_id: str):
    print(f"\n[Tool Executed] Gathering sensor data for: {location_id}...")
    return json.dumps({
        "moisture_percent": 32.5,
        "temperature_c": 21.0,
        "ph": 6.8,
        "npk": {"n_mg_kg": 18, "p_mg_kg": 12, "k_mg_kg": 15}
    })

def get_crop_requirements(crop_name: str):
    print(f"\n[Tool Executed] Checking knowledge base for: {crop_name}...")
    if crop_name.lower() == 'tomato':
        return json.dumps({
            "stage": "vegetative",
            "optimal_ph_range": [6.2, 7.0],
            "target_moisture_range": [40, 60],
            "target_npk_ratio": "3-1-2"
        })
    return json.dumps({"error": "Crop not found."})

# ==========================================
# Step 3. THE TEXT-BASED SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are a helpful agricultural assistant. 
You have access to two tools to help answer user questions:

1. "get_soil_data": Gets the latest soil condition data for a field. 
   Required arguments: {"location_id": "name of the location"}
2. "get_crop_requirements": Finds the ideal soil conditions for a specific crop. 
   Required arguments: {"crop_name": "name of the crop"}

IF YOU NEED TO USE A TOOL, you MUST reply with ONLY a JSON object in this exact format, nothing else:
{"tool": "tool_name", "arguments": {"arg_name": "value"}}

If you have enough information to answer the user fully, just write your normal response.
"""

# ==========================================
# Step 4. THE CUSTOM ORCHESTRATION LOOP
# ==========================================
def run_text_based_agent(user_query):
    print(f"User Request: {user_query}\n")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    while True:
        # UPDATE: We added the max_completion_tokens parameter exactly as the dashboard requires!
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=16384 
        )
        
        reply_text = response.choices[0].message.content.strip()
        
        # Parse the AI's reply to see if it wants to use a tool
        try:
            ai_decision = json.loads(reply_text)
            
            if "tool" in ai_decision:
                tool_name = ai_decision["tool"]
                arguments = ai_decision.get("arguments", {})
                
                messages.append({"role": "assistant", "content": reply_text})
                
                # Execute the matched tool
                if tool_name == "get_soil_data":
                    result = get_soil_data(arguments.get("location_id"))
                elif tool_name == "get_crop_requirements":
                    result = get_crop_requirements(arguments.get("crop_name"))
                else:
                    result = json.dumps({"error": "Unknown tool requested."})
                
                # Feed the data back to the AI
                messages.append({
                    "role": "system", 
                    "content": f"Result from {tool_name}: {result}"
                })
                
                print("[Agent] Reading tool data and thinking again...")
                continue 
                
        except json.JSONDecodeError:
            # If the reply is NOT JSON, it is the final answer
            print("\n--- Final Agent Advisory ---")
            print(reply_text)
            break 

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    query = "Please check the state of the tomato crop in the Cateel farm location and provide a fertilizer recommendation based on its needs."
    try:
        run_text_based_agent(query)
    except Exception as e:
        print(f"\nSystem Error: {e}")