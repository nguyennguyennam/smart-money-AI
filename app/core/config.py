from pydantic import BaseSettings

class Settings (BaseSettings):
    APP_NAME: str = "SmartMoney AI"
    VERSION: str = "1.0.0"

    OCR_MODEL_NAME = "pytesseract"
    ASR_MODEL_NAME = "whisper"

    class Config:
        env_file = ".env"
        file_encoding = "utf-8"

settings = Settings()