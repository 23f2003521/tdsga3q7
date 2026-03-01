import os
import time
import uuid
import yt_dlp
import google.generativeai as genai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# =============================
# CONFIG
# =============================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Request & Response Schema
# =============================

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


# =============================
# Utility: Download Audio Only
# =============================

def download_audio(video_url: str) -> str:
    filename = f"audio_{uuid.uuid4()}.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return filename


# =============================
# Utility: Upload to Gemini
# =============================

def upload_and_wait(filepath: str):
    uploaded_file = genai.upload_file(filepath)

    # Poll until ACTIVE
    while uploaded_file.state.name != "ACTIVE":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    return uploaded_file


# =============================
# Structured Output Schema
# =============================

response_schema = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "string",
            "description": "Timestamp in HH:MM:SS format only"
        }
    },
    "required": ["timestamp"]
}


# =============================
# Main Endpoint
# =============================


@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    # Step 1: Download audio
    audio_file = download_audio(request.video_url)

    try:
        # Step 2: Upload to Gemini Files API
        gemini_file = upload_and_wait(audio_file)

        # Step 3: Ask Gemini
        model = genai.GenerativeModel("gemini-1.5-pro")

        prompt = f"""
        The uploaded file contains audio from a YouTube video.

        Identify the FIRST time the following topic or spoken phrase appears:
        "{request.topic}"

        Return ONLY the timestamp in strict HH:MM:SS format.
        """

        response = model.generate_content(
            [prompt, gemini_file],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        result = response.json()

        return {
            "timestamp": result["timestamp"],
            "video_url": request.video_url,
            "topic": request.topic
        }

    finally:
        # Step 4: Cleanup temp file
        if os.path.exists(audio_file):
            os.remove(audio_file)
