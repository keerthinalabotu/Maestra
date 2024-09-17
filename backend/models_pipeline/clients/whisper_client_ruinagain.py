from optimum.intel import OVQuantizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
import torch
from torch.utils.data import DataLoader
import numpy as np
from openvino.runtime import Core
import nncf
from nncf.parameters import ModelType
import openvino as ov


import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)

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

# Load model and processor
model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

model.eval()

# Prepare dataset for calibration
print("Loading dataset...")
librispeech = load_dataset("librispeech_asr", "clean", split="train.360[:100]", trust_remote_code=True)
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=3000))

# def preprocess_function(examples):
#     audio_arrays = [x["array"] for x in examples["audio"]]
#     inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
#     return {"input_features": inputs.input_features}

  

calibration_dataset = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names)
# validation_dataset = calibration_dataset  # For simplicity, using the same dataset for validation

# @torch.no_grad()
# def conv_transpose_stride2(x: torch.Tensor) -> torch.Tensor:
#     dilate = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=1, stride=2, groups=128, bias=False)
#     dilate.weight.data.fill_(1.0)
#     return dilate(x)

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
quantizer = OVQuantizer.from_pretrained(model, task="automatic-speech-recognition", library="--library transformers")


def custom_export(model, config, device_for_inputs):
    dummy_input = torch.randn(1, 80, 3000, device=device_for_inputs)  # Adjust size as needed
    torch.onnx.export(model,
                      dummy_input,
                      f=config.export_output,
                      input_names=['input_features'],
                      output_names=['output'],
                      dynamic_axes={'input_features': {0: 'batch_size', 2: 'sequence'},
                                    'output': {0: 'batch_size', 1: 'sequence'}},
                      do_constant_folding=True,
                      opset_version=13)

# Quantize the model
quantized_model = nncf.quantize(
    save_directory="./quantized_whisper_model_small",
    calibration_dataset=calibration_dataset,
    # validation_dataset=validation_dataset,
    # validation_fn=validate,
    batch_size=8,
    weights_only=False,
    # device ="npu",
    library_name="transformers",
    # export_function=custom_export,
    # max_drop=0.01,
)

print("Model quantized and saved successfully!")

# Test the quantized model
test_audio = librispeech[0]["audio"]["array"]

processed_test_input = processor(
    test_audio,
    return_tensors="np",
    sampling_rate=16000,
)

if processed_test_input.input_features.shape[-1] < 3000:
    processed_test_input = processor(
        test_audio,
        return_tensors="np",
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