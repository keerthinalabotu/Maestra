from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from models_pipeline import vector_store
# import whisper
import tempfile
import os
from models_pipeline import input_processor_chunking_audio, vector_store
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from io import BytesIO
from models_pipeline.rag_testing import Generation
from text_to_speech import text_to_speech, extract_response
from fastapi.responses import JSONResponse
import logging
import io
import pydub
from pydub import AudioSegment
# import tempfile
import traceback

from fastapi.staticfiles import StaticFiles


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = 'uploads/'
generation = Generation()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
# chroma_client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_db"
# ))

# store = vector_store.VectorStore()
# collection = store.get_or_create_collection("CivilWar", "history")

# Initialize Whisper model
# model = whisper.load_model("base")

model_path = "./models_pipeline/clients/whisper_model_small"
processor_path = ".//models_pipeline/clients/whisper_processor_small"
  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
whisper_processor = WhisperProcessor.from_pretrained(processor_path)

processor = input_processor_chunking_audio.ProcessInput(whisper_model, whisper_processor)

class ProcessRequest(BaseModel):
    filepath: str

class ResponseRequest(BaseModel):
    question: str
    answer: str

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    text: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), topic: str = Form(...)):
    print(f"Received file: {file.filename}")
    print(f"Topic: {topic}")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    generation.storage.add_pdf_file_to_collection(filepath, topic)

    initial_prompt = generation.initialize_conversation(topic)
    print("hello???")
    print("Initial prompt from generation:", initial_prompt)
    extracted_response = extract_response(initial_prompt)

    if extracted_response is None:
        raise HTTPException(status_code=500, detail="Failed to extract response")

    return JSONResponse(content={
        "message": "File processed successfully",
        "file_path": filepath,
        "initial_prompt": initial_prompt,
        "extracted_response": extracted_response,
        # "audio_file": audio_url
    }, status_code=200)
    # return JSONResponse(content={"message": "File uploaded and processed successfully", "filepath": filepath}, status_code=200)

    # return JSONResponse(content={"message": "File uploaded successfully", "filepath": filepath}, status_code=200)


@app.post("/conversation_initial")
async def process_file(request: Request):
    # body = await request.json()
    # extracted_response = body.get("extracted_response")
    # logger.info(f"Received process request with filepath: {request.filepath}")

    # filepath = request.filepath
    # if not os.path.exists(filepath):
    #     raise HTTPException(status_code=400, detail="File not found")

    # # Update vector database with file contents
    # generation.storage.add_pdf_file_to_collection(filepath, "CivilWar","History")
    # # generation.storage.get_or_create_collection("CivilWar","History")

    # # Generate initial response
    # initial_prompt = generation.initialize_conversation("CivilWar-History")
    # print("hello???")
    # print("Initial prompt from generation:", initial_prompt)
    # extracted_response = extract_response(initial_prompt)

    # if extracted_response is None:
    #     raise HTTPException(status_code=500, detail="Failed to extract response")

    # Convert text to speech
    # audio_file = text_to_speech(initial_prompt)
    print("conversation_initial")
    body = await request.json()
    print(body)
    extracted_response = body.get("extracted_response")
    print('----------------------')
    print(extracted_response)
    

    if not extracted_response:
        raise HTTPException(status_code=400, detail="No extracted response provided")
    audio_file = text_to_speech(extracted_response)
    print("byeee???")
    relative_path = os.path.relpath(audio_file, "static")
    print(relative_path)
    # Construct the URL
    audio_url = f"http://localhost:8000/static/{relative_path}"
    print(audio_url)

    return JSONResponse(content={
        "message": "audio created successfully",
        # "initial_prompt": initial_prompt,
        # "extracted_response": extracted_response,
        "audio_file": audio_url
    }, status_code=200)

# @app.post("/conversation_initial")



@app.post("/question", response_model=Answer)
async def get_answer(question: Question):
    # Here you would integrate with your AI model to generate an answer
    # For now, we'll just echo the question
    return Answer(text=f"Echoing your question: {question.text}")


class TranscriptionResponse(BaseModel):
    transcription: str

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:

        temp_file_path = f"/tmp/{file.filename}"

        print(temp_file_path)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        audio = AudioSegment.from_file(temp_file_path, format=file.filename.split('.')[-1])
        wav_file_path = temp_file_path.replace(f".{file.filename.split('.')[-1]}", ".wav")
        audio.export(wav_file_path, format="wav")

        # Process with Whisper
        with open(wav_file_path, "rb") as wav_file:
            transcription = processor.process_speech(wav_file)

        # Convert webm to mp3 using pydub (with ffmpeg installed)
        # audio = AudioSegment.from_file(temp_file_path, format="webm")
        # mp3_file_path = temp_file_path.replace(".webm", ".mp3")
        # audio.export(mp3_file_path, format="mp3")


        # print(f"Received file: {file.filename}")
        # print(f"Content type: {file.content_type}")
        # audio_bytes = await file.read()

        # buffer = io.BytesIO(audio_bytes)
        # buffer.name = file.filename + ".mp3"

        # print(f"File size: {buffer.name} bytes")
        # audio_buffer = BytesIO(audio_bytes)
        
        # Transcribe audio using Whisper
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        #     temp_file.write(audio_bytes)
        #     temp_file_path = temp_file.name

        # print(f"Saved temp file: {temp_file_path}")

        # Convert webm to wav
        # audio = AudioSegment.from_file(temp_file_path, format="webm")
        # wav_path = temp_file_path.replace(".webm", ".wav")
        # audio.export(wav_path, format="wav")
        # print(f"Converted to WAV: {wav_path}")

        # # Process with Whisper
        # with open(wav_path, "rb") as wav_file:
        #     transcription = processor.process_speech(wav_file)



        # transcription = processor.process_speech(audio)


        # transcription = processor.process_speech(buffer)


        # transcription = processor.process_speech(io.BufferedReader(BytesIO(audio_bytes)))
        print(transcription)
        
        return TranscriptionResponse(transcription=transcription)
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)