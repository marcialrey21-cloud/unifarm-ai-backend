import os
import requests
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from supabase import create_client, Client
from scipy.optimize import linprog

# ==========================================
# 1. DATABASE CONFIGURATION (SUPABASE)
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. SERVER & DATA MODELS
# ==========================================
app = FastAPI(title="Uni-Farm Hub Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class FarmerQuery(BaseModel):
    message: str
    image_base64: Optional[str] = None

# NEW: Model to catch the data sent from the mobile app's Formulate screen
class FormulateQuery(BaseModel):
    target_protein: str
    ingredients: str

@app.get("/")
async def home():
    return {"message": "Welcome to the Uni-Farm Hub API! The server is online."}

# ==========================================
# ENDPOINT: Fetch Inventory
# ==========================================
@app.get("/api/inventory")
async def get_inventory():
    """Fetches the current list of feed ingredients and prices from Supabase."""
    try:
        # We select just the name, category, and price to show in the app
        response = supabase.table('feed_ingredients').select('name, category, price_php').execute()
        
        if response.data:
            return {"status": "success", "data": response.data}
        else:
            return {"status": "empty", "data": []}
            
    except Exception as e:
        print(f"[Error fetching inventory] -> {e}")
        return {"status": "error", "message": str(e)}

# ==========================================
# 3. TOOL DEFINITIONS (The "Hands" of the AI)
# ==========================================

@tool
def get_weather_data(location: str) -> dict:
    """Fetch real-time weather. Use this when the user asks about the weather."""
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {"temperature_c": data["main"]["temp"], "description": data["weather"][0]["description"]}
        return {"error": "Location not found."}
    except Exception as e:
        return {"error": str(e)}

@tool
def calculate_scientific_feed(
    target_cp: float, 
    min_energy_kcal: float, 
    min_calcium: float, 
    min_lysine: float, 
    total_kg: float, 
    ingredient_names: List[str]
) -> dict:
    """
    Calculate a SCIENTIFIC least-cost formulation.
    """
    print(f"\n[Tool] -> Scientific Formulation: {total_kg}kg | CP:{target_cp}% | ME:{min_energy_kcal}kcal | Ca:{min_calcium}% | Lys:{min_lysine}%")
    
    try:
        core_balancers = ['Limestone', 'MDCP (Monodicalcium Phos)', 'Salt', 'Vitamin-Mineral Premix', 'L-Lysine HCL', 'DL-Methionine']
        full_ingredient_list = list(set(ingredient_names + core_balancers))
        
        response = supabase.table('feed_ingredients').select('*').in_('name', full_ingredient_list).execute()
        db_ingredients = response.data
        
        if not db_ingredients:
            return {"error": "Could not connect to database or find ingredients."}

        ingredients = [item for item in db_ingredients if item['name'] in full_ingredient_list]

        names = [i['name'] for i in ingredients]
        prices = [float(i['price_php']) for i in ingredients]
        cps = [float(i['cp_percent']) for i in ingredients]
        energies = [float(i['energy_kcal_kg']) for i in ingredients]
        calciums = [float(i['calcium_percent']) for i in ingredients]
        lysines = [float(i['lysine_percent']) for i in ingredients]
        max_inclusions = [float(i['max_inclusion_percent']) / 100.0 for i in ingredients] 

        num_ing = len(ingredients)

        attempts = [
            {"cp": target_cp, "energy": min_energy_kcal, "note": "Optimal Scientific Formulation"},
            {"cp": target_cp * 0.95, "energy": min_energy_kcal * 0.95, "note": "Relaxed Formulation (Slightly lower CP & Energy)"},
            {"cp": target_cp * 0.90, "energy": min_energy_kcal * 0.90, "note": "Practical Field Formulation (Lower CP & Energy to fit ingredients)"}
        ]

        winning_res = None
        winning_attempt = None

        for attempt in attempts:
            c = prices 
            A_eq = [[1] * num_ing]
            b_eq = [1]
            A_ub = [
                [-cp for cp in cps],               
                [-energy for energy in energies],  
                [-ca for ca in calciums],          
                [-lys for lys in lysines]          
            ]
            b_ub = [
                -attempt["cp"], 
                -attempt["energy"], 
                -min_calcium, 
                -min_lysine
            ]
            bounds = [(0, max_inclusions[i]) for i in range(num_ing)]

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                winning_res = res
                winning_attempt = attempt
                break 

        if not winning_res:
             return {"error": "Math is impossible even with relaxed constraints. You MUST include high-energy sources like 'Crude Palm Oil' or high-protein like 'Soybean Meal (44%)' to make this work."}

        recipe = {}
        total_cost = 0
        actual_cp = actual_energy = actual_ca = actual_lys = 0
        fractions = winning_res.x

        for i in range(num_ing):
            kg_needed = round(fractions[i] * total_kg, 2)
            cost_for_ing = round(kg_needed * prices[i], 2)
            
            actual_cp += fractions[i] * cps[i]
            actual_energy += fractions[i] * energies[i]
            actual_ca += fractions[i] * calciums[i]
            actual_lys += fractions[i] * lysines[i]

            if kg_needed > 0.05: 
                 recipe[names[i]] = {"kg": kg_needed, "cost_php": cost_for_ing}
                 total_cost += cost_for_ing

        return {
            "success": True,
            "status": winning_attempt["note"],
            "total_batch_kg": total_kg,
            "total_cost_php": round(total_cost, 2),
            "achieved_nutrition": {
                "CP_%": round(actual_cp, 2),
                "Energy_kcal": round(actual_energy, 0),
                "Calcium_%": round(actual_ca, 2),
                "Lysine_%": round(actual_lys, 2)
            },
            "recipe": recipe,
        }

    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}

# ==========================================
# 4. AI AGENT CONFIGURATION
# ==========================================

farm_tools = [get_weather_data, calculate_scientific_feed]

# Fixed Syntax Error Here
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

llm = AzureChatOpenAI(
    azure_endpoint="https://cropadvisoragent-resource.cognitiveservices.azure.com/",
    azure_deployment="unifarm-agent-v1",
    openai_api_version="2024-12-01-preview", 
    api_key=AZURE_API_KEY,
    temperature=0.1 
)

unifarm_agent = create_react_agent(model=llm, tools=farm_tools)

system_instruction = SystemMessage(content=(
    "You are the Uni-Farm Hub AI Orchestrator, an expert in Animal Nutrition and Agronomy in the Philippines. "
    "When a user asks to formulate feed, you must intelligently decide the target nutritional requirements (CP, Energy kcal, Calcium, Lysine) based on the animal they specify. "
    "Select a logical list of ingredients from the database. "
    
    "CRITICAL UX RULE: NEVER ask the user for 'exact database names', strings, or table schemas. A farmer does not know this. "
    "Map common requests silently using this dictionary before calling tools: "
    "- 'Corn' -> 'Yellow Corn' "
    "- 'Rice Bran' -> 'Rice Bran (D1)' "
    "- 'Soybean' -> 'Soybean Meal (44%)' "
    "- 'Fish Meal' -> 'Fish Meal (Local 55%)' "
    "- 'Copra' -> 'Copra Meal' "
    "- 'Limestone' -> 'Limestone' "
    "- 'MDCP' or 'Phosphorus' -> 'MDCP (Monodicalcium Phos)' "
    "- 'Salt' -> 'Salt' "
    "- 'Premix' or 'Vitamins' -> 'Vitamin-Mineral Premix' "
    "- 'Lysine' -> 'L-Lysine HCL' "
    "- 'Methionine' -> 'DL-Methionine' "
    
    "Note: The calculate_scientific_feed tool automatically injects balancing minerals in the background. Do not halt the conversation to ask for string confirmations. Just run the tool and present the final recipe. "
    
    "CRITICAL FORMATTING RULE: ALWAYS use English or Taglish for your final response. NEVER use Arabic or other foreign alphabets. When formatting currency, strictly use 'PHP' or the '₱' symbol (e.g., '₱24.12 per kg'). Do not add translated words to numbers."
))

# ==========================================
# 5. THE API ENDPOINTS
# ==========================================

# NEW: The dedicated endpoint for the mobile app's Formulate screen!
@app.post("/api/formulate")
async def formulate_feed_api(query: FormulateQuery):
    try:
        # We build a prompt for your agent using the data sent from the phone
        agent_prompt = f"""
        A farmer has requested a feed formulation.
        Target Crude Protein: {query.target_protein}%
        Available Ingredients: {query.ingredients}
        
        Please use your calculate_scientific_feed tool to formulate a 100kg batch based on these parameters. 
        Once the tool succeeds, present the final scientific recipe cleanly so it looks great on a mobile screen.
        """
        
        # Send it to your existing, brilliant agent
        user_message = HumanMessage(content=agent_prompt)
        response = unifarm_agent.invoke({"messages": [system_instruction, user_message]})
        final_message = response["messages"][-1].content
        
        return {"recipe": final_message}
    except Exception as e:
        print(f"[Formulate Error] -> {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_agent(query: FarmerQuery):
    try:
        if query.image_base64:
            clean_base64 = query.image_base64
            if "base64," in clean_base64:
                clean_base64 = clean_base64.split("base64,")[-1]
            clean_base64 = clean_base64.replace("\n", "").replace("\r", "").replace(" ", "")
            
            advanced_expert_prompt = """
            You are a leading agricultural pathologist.

            First, provide a 'Quick Summary' (2 to 3 sentences maximum) stating the most likely plant, the most likely disease, and the immediate recommended action.
            Then, you MUST insert this exact text on a new line: ---DETAILS---
            Finally, below that delimiter, provide your comprehensive, step-by-step analysis:
            1. PHASE 1: IMAGE VERIFICATION.
            2. PHASE 2: BOTANICAL CLASSIFICATION.
            3. PHASE 3: DISEASE DIAGNOSIS.
            4. PHASE 4: Format clearly.
            """
            
            multimodal_content = [
                {"type": "text", "text": advanced_expert_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}"}}
            ]
            user_message = HumanMessage(content=multimodal_content)
            response = llm.invoke([user_message])
            final_message = response.content
            
        else:
            user_message = HumanMessage(content=query.message)
            response = unifarm_agent.invoke({"messages": [system_instruction, user_message]})
            final_message = response["messages"][-1].content
        
        return {"status": "success", "reply": final_message}
        
    except Exception as e:
        print(f"[Fatal Error] -> {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# NEW ENDPOINT: Advanced AI Predictor
# ==========================================

# Data model to catch the diverse inputs from the React Native app
class PredictAdvancedQuery(BaseModel):
    category: str
    temp_celsius: float
    area_hectares: Optional[float] = 0.0
    rainfall_mm: Optional[float] = 0.0
    fertilizer_kg: Optional[float] = 0.0
    head_count: Optional[int] = 0
    feed_kg: Optional[float] = 0.0

@app.post("/api/predict/advanced")
async def predict_advanced(query: PredictAdvancedQuery):
    try:
        math_yield_text = ""
        prompt_context = ""
        
        # 1. The Mathematical Baseline (Deterministic)
        if query.category in ['corn', 'rice', 'vegetables']:
            # Assuming a standard average of ~4.5 tons per hectare as a baseline
            base_yield = float(query.area_hectares) * 4.5
            # Simple math modifiers based on inputs
            if query.rainfall_mm < 50:
                base_yield *= 0.7  # Drought penalty
            if query.fertilizer_kg > 100:
                base_yield *= 1.1  # Fertilizer bonus
                
            predicted_tons = round(base_yield, 2)
            math_yield_text = f"{predicted_tons} Tons"
            prompt_context = f"Crop: {query.category}, Area: {query.area_hectares}ha, Rainfall: {query.rainfall_mm}mm, Fertilizer: {query.fertilizer_kg}kg, Temp: {query.temp_celsius}°C."
            
        else:
            # For Livestock, Poultry, Tilapia
            # Assuming survival rate math based on basic parameters
            survival_rate = 0.90
            if query.temp_celsius > 32:
                survival_rate = 0.80 # Heat stress penalty
                
            surviving_heads = int(float(query.head_count) * survival_rate)
            math_yield_text = f"{surviving_heads} Heads (Est. Survival)"
            prompt_context = f"Animals: {query.category}, Starting Heads: {query.head_count}, Feed Available: {query.feed_kg}kg, Temp: {query.temp_celsius}°C."

        # 2. The Azure OpenAI Generative Insight
        agent_prompt = f"""
        You are an expert agronomist for Uni-Farm Hub in the Philippines.
        I have mathematically calculated the baseline yield for the following scenario:
        Context: {prompt_context}
        Mathematical Yield: {math_yield_text}
        
        Provide a brief, professional 'AI Insight' (2-3 sentences max) to advise the farmer on this scenario. 
        Focus on the temperature, feed/fertilizer adequacy, or rainfall risks. 
        Do NOT write a greeting, just provide the insight directly.
        """
        
        user_message = HumanMessage(content=agent_prompt)
        # Use your existing llm connection defined earlier in api.py
        response = llm.invoke([user_message])
        ai_advice = response.content
        
        # 3. Return the hybrid payload back to the mobile app
        return {
            "predicted_yield": math_yield_text,
            "ai_insight": ai_advice
        }

    except Exception as e:
        print(f"[Prediction Error] -> {e}")
        raise HTTPException(status_code=500, detail=str(e))        