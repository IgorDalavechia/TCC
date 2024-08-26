from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import fastai
from fastai.data.all import *
from fastai.vision.all import *
from pydantic import BaseModel
import uvicorn
from fastai.vision.all import load_learner
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregue o modelo Fastai
path = "/home/pc/Documents/tcc/New_results/Resnet34/Epoch25/model.pkl"
learn = load_learner(path)

class PredictionResponse(BaseModel):
    class_name: str
    accuracy: float
    

@app.post("/processar_imagem/")
async def processar_imagem(file: UploadFile = File(...)):
    try:
        # Leitura da imagem
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        
      
        # Previsão
        
        pred, pred_idx, probs = learn.predict(image)

        # Obtendo a classe e a acurácia
        class_name = pred
        accuracy = probs[pred_idx].item()
        
        
            
        if(accuracy <= 0.64):
        	accuracy = 0
        	class_name= "Classe não registrada"

        response = PredictionResponse(class_name=class_name, accuracy=accuracy)
        return JSONResponse(content=response.dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
  
