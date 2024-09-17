import PyPDF2
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np

# model_path = "./clients/whisper_model_small"
# processor_path = "./clients/whisper_processor_small"

# whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
# whisper_processor = WhisperProcessor.from_pretrained(processor_path)

class ProcessInput: 
    def __init__(self, whisper_model, whisper_processor):
        self.whisper_model = whisper_model
        self.whisper_processor = whisper_processor

    def process_pdf(self, file_path):

        input_content=''

        try: 
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    input_content += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
    
        return input_content
    
    def process_speech(self, audio_file):
        try:
            # audio_input = librosa.load(audio_file, sr=16000,return_tensors="pt").input_features

            audio, sr = librosa.load(audio_file, sr=16000)

            audio = audio.astype(np.float32)
            input_features = self.whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription
            # return result["text"]
        except Exception as e:
            print(f"Error processing speech: {e}")
            return ""

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
