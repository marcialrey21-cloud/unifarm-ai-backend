from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your Expo web app to talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down to your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the exact data structure we expect from the mobile app
class CropData(BaseModel):
    crop_type: str
    area_hectares: float
    rainfall_mm: float
    fertilizer_kg: float
    temp_celsius: float

@app.post("/api/predict/crop")
async def predict_crop_yield(data: CropData):
    # --- THE TRACER BULLET MODEL ---
    # This is a smart heuristic formula. Later, we replace these 5 lines 
    # with `prediction = model.predict([data.area, data.rainfall...])`
    
    base_yield_per_ha = 4.5 if data.crop_type.lower() == 'corn' else 3.8
    
    # Simple multipliers based on optimal conditions
    rain_factor = 1.0 - abs(100 - data.rainfall_mm) / 200  # Prefers ~100mm
    temp_factor = 1.0 - abs(28 - data.temp_celsius) / 40   # Prefers ~28C
    fert_factor = 1.0 + (data.fertilizer_kg / 1000)        # Diminishing returns in real life, simple linear here
    
    # Calculate final yield in Metric Tons
    predicted_tons = (base_yield_per_ha * data.area_hectares) * rain_factor * temp_factor * fert_factor
    
    # Ensure it doesn't drop below 0
    final_prediction = max(0.1, round(predicted_tons, 2))
    
    return {
        "status": "success",
        "crop": data.crop_type,
        "predicted_yield_tons": final_prediction,
        "message": f"Optimal conditions analyzed for {data.area_hectares} hectares."
    }

# Run this with: uvicorn main:app --reload