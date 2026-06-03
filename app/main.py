from fastapi import FastAPI
from route.endpoints import router

app = FastAPI(
    title="OCR API",
    version="1.0.0"
)

app.include_router(router, prefix="/api")