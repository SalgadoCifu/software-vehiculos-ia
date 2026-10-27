import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

API_KEY = os.getenv("API_KEY")
app = FastAPI(title="Servidor IA - Vehículos")

model = YOLO("modelo/best.pt")  # se carga UNA sola vez al arrancar

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detectar")
def detectar(imagen: UploadFile = File(...), x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")

    img = Image.open(imagen.file).convert("RGB")
    resultados = model.predict(img, imgsz=640, conf=0.25, verbose=False)

    detecciones = []
    for r in resultados:
        alto, ancho = r.orig_shape
        for i, box in enumerate(r.boxes):
            det = {
                "clase": r.names[int(box.cls)],
                "confianza": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            }
            if r.masks is not None:
                poligono = r.masks.xy[i].astype(np.float32)
                area = cv2.contourArea(poligono) if len(poligono) >= 3 else 0
                det["area_pct"] = round(float(area) / (ancho * alto) * 100, 2)
            detecciones.append(det)
    return {"detecciones": detecciones}