from optimum.intel import OVQuantizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
import nncf
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import openvino as ov

def create_validation_loader(validation_dataset, batch_size=8):
    return DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


def calculate_wer(pred, ref):
    # Implement Word Error Rate calculation
    # This is a simplified implementation and might need to be improved
    # for production use (e.g., handling insertions and deletions)
    errors = sum(1 for p, r in zip(pred, ref) if p != r)
    total_words = len(ref)
    return errors / total_words if total_words > 0 else 0

def validate(model, validation_loader):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in validation_loader:
            input_features = batch["input_features"].to(model.device)
            
            # Generate predictions
            output = model.generate(input_features=input_features)
            transcriptions = processor.batch_decode(output, skip_special_tokens=True)
            
            # Get ground truth transcriptions
            ground_truth = [sample["text"] for sample in batch["original_data"]]
            
            # Convert transcriptions to word lists
            pred_words = [t.split() for t in transcriptions]
            ref_words = [t.split() for t in ground_truth]
            
            predictions.extend(pred_words)
            references.extend(ref_words)
    
    total_wer = sum(calculate_wer(pred, ref) for pred, ref in zip(predictions, references))
    avg_wer = total_wer / len(predictions)

    return 1 - avg_wer

# Load model and processor
model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

# Prepare dataset for calibration
print("Loading dataset...")
librispeech = load_dataset("librispeech_asr", "clean", split="train.360[:100]", trust_remote_code=True)
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
    return {"input_features": inputs.input_features}



validation_dataset = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names)
validation_loader = create_validation_loader(validation_dataset)


calibration_dataset = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names)

# Initialize quantizer
# quantizer = OVQuantizer.from_pretrained(model)

# nncf_config = nncf.QuantizationConfig(
#     input_info={"input_features": {0: 1, 1: 80, 2: -1}},  # Dynamic last dimension
#     compression=nncf.CompressionConfig(
#         algorithm="quantization",
#         activations=nncf.ActivationQuantizationConfig(mode="symmetric"),
#         weights=nncf.WeightQuantizationConfig(mode="symmetric", per_channel=True)
#     )
# )


quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset=calibration_dataset,
    # validation_dataset=validation_dataset,
    # validation_fn=validate,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)

print("Model quantized and saved successfully!")
model_int8 = ov.compile_model(quantized_model)
print("Model compiled with OpenVINO successfully!")



# Test the quantized model
test_audio = librispeech[0]["audio"]["array"]
input_features = processor(test_audio, sampling_rate=16000, return_tensors="np").input_features

input_fp32 = {model_int8.input(0): input_features}

res = model_int8(input_fp32)

output = list(res.values())[0]  # Assuming single output
transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")

# output = quantized_model.generate(input_features=input_features)
# transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
# print(f"Transcription: {transcription}")