from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import torch
# import intel_extension_for_pytorch as ipex
from openvino.runtime import Core
import numpy as np

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CPU available: {torch.device('cpu').type}")

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

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("./whisper_model_small")
processor = WhisperProcessor.from_pretrained("./whisper_processor_small")
audio_arrays = [audio["array"] for audio in examples["audio"]]
print(f"Number of audio samples: {len(audio_arrays)}")

if audio_arrays:
    print(f"Sample audio array shape: {audio_arrays[0].shape}")

    processed_inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        # truncation=False,
        truncation=True,
        padding="max_length",
        # padding=True,
        max_length=3000,
        return_attention_mask=True,
    )
    print(f"Shape of input_features: {processed_inputs.input_features.shape}")

    if processed_inputs.input_features.shape[-1] < 3000:
        # Re-process for short-form audio
        processed_inputs = processor(
            audio_arrays,
            return_tensors="pt",
            sampling_rate=16000,
        )
    
    # decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * len(audio_arrays))

    
    # Process text
    text_inputs = processor.tokenizer(examples["text"], padding="longest", return_tensors="pt")
        
    input_features = processed_inputs.input_features

    if len(input_features.shape) == 2:
        input_features = np.expand_dims(input_features, axis=1)
    elif len(input_features.shape) == 3:
        input_features = np.transpose(input_features, (0, 2, 1))

    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * len(audio_arrays))

    # return {
    #     "input_features": processed_inputs.input_features,
    #     "labels": text_inputs.input_ids,
    #     "decoder_input_ids": decoder_input_ids
    # }

# Load and process the dataset
# librispeech = load_dataset("librispeech_asr", "clean", split="train[:1%]")  # Using only 1% of the data for faster processing
common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train[:1%]", streaming=True, trust_remote_code=True)  # Adjust the split size as needed
print("Dataset...")
print(common_voice[0])  # Print the first example
print("--------------")
print(common_voice.column_names)  # Print the column names

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))




dataset = common_voice.map(preprocess_function, remove_columns=common_voice.column_names, cache_file_name="cached_commonvoice", batched=True)

# Set the format of the dataset
dataset.set_format(type="torch", columns=["input_features", "labels"])

# Create DataLoader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
device = torch.device("gpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_features, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the model
model.save_pretrained("./whisper_model_small_commonVoice_Libiri")
processor.save_pretrained("./whisper_processor_small_commonVoice_Libiri")