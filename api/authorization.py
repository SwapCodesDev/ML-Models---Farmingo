import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import HTTPException, Header

api_key=os.getenv("FARMINGO_API_KEY")

def validate_api_key(x_api_key: str = Header(None)):
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
