from optimum.intel import OVQuantizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
import torch
from torch.utils.data import DataLoader
import numpy as np

# Load model and processor
model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)


# Prepare dataset for calibration
print("Loading dataset...")
librispeech = load_dataset("librispeech_asr", "clean", split="train.360[:100]", trust_remote_code=True)
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=3000))

# def preprocess_function(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
#     return {"input_features": inputs.input_features}

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    # Process inputs with padding to the longest in the batch
    processed_inputs = processor(
        audio_arrays,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=16000,
    )
    
    # Check if we have short-form audio (less than 3000 frames)
    if processed_inputs.input_features.shape[-1] < 3000:
        # Re-process for short-form audio
        processed_inputs = processor(
            audio_arrays,
            return_tensors="pt",
            sampling_rate=16000,
        )
    
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * len(audio_arrays))

    return {
        "input_features": processed_inputs.input_features,
        "decoder_input_ids": decoder_input_ids
    }


calibration_dataset = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names)
# validation_dataset = calibration_dataset  # For simplicity, using the same dataset for validation

def create_validation_loader(validation_dataset, batch_size=8):
    return DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# validation_loader = create_validation_loader(validation_dataset)

def validate(model, validation_loader):
    model.eval()
    total_wer = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            input_features = batch["input_features"]
            output = model.generate(input_features=input_features)
            transcriptions = processor.batch_decode(output, skip_special_tokens=True)
            
            # Dummy WER calculation (replace with actual WER calculation in production)
            total_wer += len(transcriptions)
            total_samples += len(transcriptions)
    
    avg_wer = total_wer / total_samples
    return 1 - avg_wer  # Return accuracy (1 - WER)

# Initialize quantizer
quantizer = OVQuantizer.from_pretrained(model, task="automatic-speech-recognition", library="transformers")

# Quantize the model
quantized_model = quantizer.quantize(
    save_directory="./quantized_whisper_model_small",
    calibration_dataset=calibration_dataset,
    # validation_dataset=validation_dataset,
    # validation_fn=validate,
    batch_size=8,
    # max_drop=0.01,
)

print("Model quantized and saved successfully!")

# Test the quantized model
test_audio = librispeech[0]["audio"]["array"]

processed_test_input = processor(
    test_audio,
    return_tensors="pt",
    sampling_rate=16000,
)

if processed_test_input.input_features.shape[-1] < 3000:
    processed_test_input = processor(
        test_audio,
        return_tensors="pt",
        sampling_rate=16000,
    )

decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])


output = quantized_model.generate(
    input_features=processed_test_input.input_features,
    decoder_input_ids=decoder_input_ids
)

transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")

# input_features = processor(test_audio, sampling_rate=3000, return_tensors="pt").input_features

# output = quantized_model.generate(input_features=input_features)
# transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
# print(f"Transcription: {transcription}")