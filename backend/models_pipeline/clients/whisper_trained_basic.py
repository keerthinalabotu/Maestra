import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from openvino.runtime import Core

ie = Core()
available_devices = ie.available_devices
print(f"Available devices: {available_devices}")

if 'NPU' in available_devices:
    device = 'NPU'
elif 'GPU' in available_devices:
    device = 'GPU'
else:
    device = 'CPU'
print(f"Using device: {device}")

# Load the dataset
print("Loading dataset...")

librispeech = load_dataset("librispeech_asr", "clean", split="train.360[:100]", trust_remote_code=True)
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=3000))

# Load the pre-trained model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")


def preprocess_function(examples):
    audio_arrays = examples["audio"]["array"]  # This should be correct for LibriSpeech
    
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


# Prepare the dataset
def prepare_dataset(batch):
    # audio = batch["audio"]
    # input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # batch["input_features"] = input_features[0]
    processed = preprocess_function(batch)
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids

    batch.update(processed)
    return batch

dataset = librispeech.map(prepare_dataset, remove_columns=librispeech.column_names)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in data_loader:
        inputs = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("./whisper_model_small")
processor.save_pretrained("./whisper_processor_small")

# # Inference on user speech (pseu do-code)
# def transcribe_user_speech():
#     user_audio = record_audio()  # Implement this function to record user audio
#     input_features = processor(user_audio, sampling_rate=16000, return_tensors="pt").input_features
    
#     with torch.no_grad():
#         predicted_ids = model.generate(input_features)
    
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#     return transcription[0]