from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ml_pipeline.fusion_predict import fusion_predict

import os
import shutil

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

templates = Jinja2Templates(directory="templates")


# ---------------- HOME PAGE ----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ---------------- SYMPTOMS PAGE ----------------

@app.get("/symptoms", response_class=HTMLResponse)
async def symptoms(request: Request):
    return templates.TemplateResponse(
        "symptoms.html",
        {"request": request}
    )


# ---------------- TREATMENT PAGE ----------------

@app.get("/treatment", response_class=HTMLResponse)
async def treatment(request: Request):
    return templates.TemplateResponse(
        "treatment.html",
        {"request": request}
    )


# ---------------- FAQ PAGE ----------------

@app.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    return templates.TemplateResponse(
        "faq.html",
        {"request": request}
    )


# ---------------- PREDICTION ----------------

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    firstname: str = Form(...),
    age: int = Form(...),
    file: UploadFile = File(...)
):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = fusion_predict(file_path)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": file.filename,
            "fn": firstname,
            "age": age,
            "r": label,
            "probab": round(confidence * 100, 2)
        },
    )