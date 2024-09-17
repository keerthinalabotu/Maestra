import PyPDF2
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
from transformers import EncoderDecoderCache
import torch

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

            print(f"Type of audio_file: {type(audio_file)}")
            
            audio, sr = librosa.load(audio_file, sr=16000)

            audio = audio.astype(np.float32)

            print(f"Audio length: {len(audio)}")

            chunk_length = 20 * 16000
            # stride_length = chunk_length // 2  # 50% overlap
            stride_length=int(chunk_length * 0.95)

            transcriptions = []

            for start in range(0, len(audio), stride_length):
                end = start + chunk_length
                chunk = audio[start:end]
            
                if len(chunk) < chunk_length:
                    chunk = np.pad(chunk, (0, chunk_length - len(chunk)), 'constant')

                inputs = self.whisper_processor(
                    chunk, 
                    return_tensors="pt", 
                    sampling_rate=16000,
                    return_attention_mask=True
                )

                input_features = inputs.input_features
                attention_mask = inputs.attention_mask

                generated_ids = self.whisper_model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language='en',
                    max_length=448,
                    num_beams=5,
                    early_stopping=True
                )

                chunk_transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(chunk_transcription)
                transcriptions.append(chunk_transcription)

            # Join all transcriptions
            full_transcription = ' '.join(transcriptions)
        
            return full_transcription





            # inputs = self.whisper_processor(
            #     audio, 
            #     return_tensors="pt", 
            #     sampling_rate=16000,
            #     return_attention_mask=True  # Explicitly request attention mask
            # )

            # input_features = inputs.input_features
            # # attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=input_features.device)
            # attention_mask = inputs.attention_mask

            # generated_ids = self.whisper_model.generate(
            #     input_features,
            #     attention_mask=attention_mask,  # Pass attention mask
            #     language='en',  # Specify English if you want to force English output
            #     # past_key_values=EncoderDecoderCache(),  # Use EncoderDecoderCache instead of tuple
            #     max_length=448  # Adjust as needed
            # )  
            # # input_features = self.whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features
            # # predicted_ids = self.whisper_model.generate(input_features)
            # transcription = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # # transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # return transcription
            # # return result["text"]
        except Exception as e:
            print(f"Error processing speech: {e}")
            return ""

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
