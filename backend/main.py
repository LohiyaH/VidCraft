from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import subprocess
from dotenv import load_dotenv
from huggingface_hub import InferenceClient  # For image generation
from elevenlabs import ElevenLabs  # For text-to-speech
from io import BytesIO  # For handling image bytes
import re  # For text cleaning
import whisper  # For subtitle generation
from storage import Storage  # Import the Storage class
import asyncio
import traceback
import uuid  # For generating random filenames
from pathlib import Path  # For handling file paths
import requests  # For downloading subtitles


# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize the Storage class
storage = Storage()

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

# ElevenLabs API configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is not set in the .env file")

# Hugging Face InferenceClient for Stable Diffusion
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is not set in the .env file")

client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=HUGGINGFACE_API_KEY)

# Load Whisper model
whisper_model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy

# Request models
class GenerateVideoRequest(BaseModel):
    topic: str  # The topic of the video
    background: str  # Description of the background for the image

# Text cleaning function
def clean_text(text):
    """
    Clean the text by removing unwanted characters and extra spaces.
    """
    # Remove special characters and extra spaces
    cleaned_text = re.sub(r"[^\w\s.,!?]", "", text)  # Keep letters, numbers, and basic punctuation
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Replace multiple spaces with a single space
    return cleaned_text.strip()

# Function to format timestamps for SRT
def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,ms).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

# Function to clean and reformat subtitles
def clean_subtitles(subtitles: str) -> str:
    """
    Clean and reformat subtitles to match the SRT format expected by FFmpeg.
    """
    cleaned_subtitles = []
    subtitle_blocks = subtitles.strip().split("\n\n")  # Split into individual subtitle blocks

    for i, block in enumerate(subtitle_blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue  # Skip invalid blocks

        # Extract start and end times
        start_time, end_time = lines[1].split(" --> ")
        start_time = format_timestamp(float(start_time))
        end_time = format_timestamp(float(end_time))

        # Extract text
        text = "\n".join(lines[2:]).strip()

        # Reformat the subtitle block
        cleaned_subtitles.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")

    return "\n".join(cleaned_subtitles)

# Function to convert timestamp to seconds
def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert a timestamp in HH:MM:SS,ms format to seconds.
    """
    try:
        hh_mm_ss, ms = timestamp.split(",")  # Split into seconds and milliseconds
        hh, mm, ss = hh_mm_ss.split(":")  # Split into hours, minutes, and seconds
        total_seconds = int(hh) * 3600 + int(mm) * 60 + float(ss) + float(ms) / 1000
        return total_seconds
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}. Expected HH:MM:SS,ms.")

# Generate subtitles using Whisper
def generate_subtitles(audio_file_path: str) -> str:
    """
    Generate subtitles in SRT format from an audio file using Whisper.
    """
    try:
        print("Generating subtitles using Whisper...")
        result = whisper_model.transcribe(audio_file_path)

        subtitles = []
        for i, segment in enumerate(result["segments"]):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            subtitles.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")

        # Clean and reformat the subtitles
        cleaned_subtitles = clean_subtitles("\n".join(subtitles))
        return cleaned_subtitles

    except Exception as e:
        print("Error generating subtitles:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint
@app.get("/api/test")
async def test():
    return {"message": "Hello, World!"}

# Video endpoint
@app.post("/api/generate-video")
async def generate_video(request: GenerateVideoRequest):
    try:
        print("Received video generation request:", request)

        # Step 1: Generate text using Gemini API
        print("Generating text using Gemini API...")
        text_response = model.generate_content(f"Generate content about: {request.topic}")
        generated_text = clean_text(text_response.text)
        print("Generated text:", generated_text)

        # Step 2: Generate audio using ElevenLabs API
        print("Generating audio using ElevenLabs API...")
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id="CwhRBWXzGAHq8TQ4Fs17",  # Replace with your voice ID
            text=generated_text,
        )
        audio_bytes = b"".join(audio_generator)

        # Save the audio file using the Storage class
        audio_file_name = f"{uuid.uuid4()}.mp3"
        audio_url = storage.save_file(audio_file_name, audio_bytes)
        print("Audio file saved:", audio_url)

        # Step 3: Generate image using Hugging Face API
        print("Generating image using Hugging Face API...")
        image = client.text_to_image(request.background)  # Generate image based on background description
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Save the image file using the Storage class
        image_file_name = f"{uuid.uuid4()}.png"
        image_url = storage.save_file(image_file_name, image_bytes)
        print("Image file saved:", image_url)

        # Step 4: Generate subtitles using Whisper
        print("Generating subtitles using Whisper...")
        subtitles = generate_subtitles(audio_url)

        # Save the subtitles using the Storage class
        subtitle_file_name = f"{uuid.uuid4()}.srt"
        subtitle_url = storage.save_file(subtitle_file_name, subtitles.encode("utf-8"))
        print("Subtitles file saved:", subtitle_url)

        # Step 5: Download all files locally
        storage_dir = Path("D:/projects_webd/shorty/backend/storage")  # Absolute path to storage folder
        storage_dir.mkdir(exist_ok=True)  # Ensure the storage directory exists

        # Download image file locally
        image_local_path = storage_dir / image_file_name
        with open(image_local_path, "wb") as f:
            response = requests.get(image_url)
            f.write(response.content)
        print("Image file downloaded locally:", image_local_path)

        # Download audio file locally
        audio_local_path = storage_dir / audio_file_name
        with open(audio_local_path, "wb") as f:
            response = requests.get(audio_url)
            f.write(response.content)
        print("Audio file downloaded locally:", audio_local_path)

        # Download subtitle file locally
        subtitle_local_path = storage_dir / subtitle_file_name
        with open(subtitle_local_path, "wb") as f:
            response = requests.get(subtitle_url)
            f.write(response.content)
        print("Subtitles file downloaded locally:", subtitle_local_path)

        # Step 6: Determine video duration from subtitles
        subtitle_lines = subtitles.strip().split("\n")
        last_subtitle_end_time = timestamp_to_seconds(subtitle_lines[-2].split(" --> ")[1])  # Convert to seconds
        video_duration = int(last_subtitle_end_time) + 1  # Add 1 second buffer
        print("Calculated video duration:", video_duration)

        # Step 7: Generate video using FFmpeg
        print("Generating video using FFmpeg...")
        video_file_name = f"{uuid.uuid4()}.mp4"
        video_file_path = storage_dir / video_file_name

        # Convert paths to absolute paths and replace backslashes with forward slashes
        image_local_path_abs = os.path.abspath(image_local_path).replace("\\", "/")
        audio_local_path_abs = os.path.abspath(audio_local_path).replace("\\", "/")
        subtitle_local_path_abs = os.path.abspath(subtitle_local_path).replace("\\", "/")
        video_file_path_abs = os.path.abspath(video_file_path).replace("\\", "/")

        # Debugging: Print absolute paths
        print("Absolute paths:")
        print("Image:", image_local_path_abs)
        print("Audio:", audio_local_path_abs)
        print("Subtitles:", subtitle_local_path_abs)
        print("Output video:", video_file_path_abs)

        # Ensure subtitle file exists
        if not os.path.exists(subtitle_local_path_abs):
            raise HTTPException(status_code=500, detail=f"Subtitle file not found: {subtitle_local_path_abs}")

        # Escape colons in the subtitle path
        subtitle_path_escaped = subtitle_local_path_abs.replace(":", "\\:")

        # FFmpeg command to combine image, audio, and subtitles
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file without asking
            "-loop", "1",  # Loop the image
            "-i", image_local_path_abs,  # Input image (absolute path)
            "-i", audio_local_path_abs,  # Input audio (absolute path)
            "-vf", f"subtitles='{subtitle_path_escaped}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&'",  # Add subtitles
            "-c:v", "libx264",  # Video codec
            "-t", str(video_duration),  # Duration of the video
            "-pix_fmt", "yuv420p",  # Pixel format
            "-shortest",  # Finish encoding when the shortest input ends
            video_file_path_abs,  # Output video (absolute path)
        ]

        # Debugging: Print the FFmpeg command
        print("FFmpeg command:", " ".join(command))

        # Run FFmpeg in a background thread
        def run_ffmpeg():
            result = subprocess.run(command, capture_output=True, text=True)
            return result

        result = await asyncio.to_thread(run_ffmpeg)

        # Print FFmpeg output for debugging
        print("FFmpeg stdout:", result.stdout)
        print("FFmpeg stderr:", result.stderr)

        if result.returncode != 0:
            error_message = f"FFmpeg command failed: {result.stderr}"
            print(error_message)
            raise HTTPException(status_code=500, detail=error_message)

        # Save the video file using the Storage class
        with open(video_file_path, "rb") as f:
            video_bytes = f.read()
        video_url = storage.save_file(video_file_name, video_bytes)

        print("Video file saved:", video_url)
        return {"videoUrl": video_url}

    except Exception as e:
        # Log the full traceback
        error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)