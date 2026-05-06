from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
import sys
import os

# Ensure ml_core is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.inference import AgeProgressor

app = FastAPI(title="Age Progression GAN API")

# Setup CORS to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AgeProgressor model
# Assuming no weights for now, it will output random noise based on the uninitialized model
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "latest-G.ckpt")
    progressor = AgeProgressor(model_path=model_path if os.path.exists(model_path) else None)
except Exception as e:
    print(f"Error initializing AgeProgressor: {e}")
    progressor = None

@app.get("/")
def read_root():
    return {"message": "Age Progression GAN API is running."}

@app.post("/api/progress_age")
async def progress_age(
    file: UploadFile = File(...),
    target_age_group: int = Form(...)
):
    if not progressor:
        return JSONResponse(status_code=500, content={"message": "Model not initialized."})
    
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Ensure age group is within bounds (0 to 5)
        target_age_group = max(0, min(5, int(target_age_group)))
        
        # Process the image
        output_image = progressor.progress_age(image, target_age_group)
        
        # Convert output image to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"status": "success", "image_base64": img_str}
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
