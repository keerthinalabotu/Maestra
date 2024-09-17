from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
from datasets import load_dataset, concatenate_datasets, Audio, config
import torch
from openvino.runtime import Core
import shutil
from nncf import quantize
import nncf
import time
from functools import partial
from openvino.tools import mo
from openvino.runtime import serialize
# from openvino.tools.pot import DataLoader, compress_model_weights

def clear_cache():
    if config.DOWNLOADED_DATASETS_PATH:
        shutil.rmtree(config.DOWNLOADED_DATASETS_PATH, ignore_errors=True)
        print(f"Cache directory cleared: {config.DOWNLOADED_DATASETS_PATH}")


# config.DOWNLOADED_DATASETS_PATH = r"C:\Users\hermana\LibriSpeech_Cache"
# clear_cache() 
 
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

# device = 'GPU' if 'GPU' in available_devices else 'CPU'
# print(f"Using device: {device}")

model_id = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)

print("Loading dataset...")
# common_voice = load_dataset("common_voice", "en", split="train[:50]", trust_remote_code=True)
librispeech = load_dataset("librispeech_asr", "clean", split="train.360[:50]", trust_remote_code=True)
# ted_lium = load_dataset("ted_lium", split="train[:50]", trust_remote_code=True)
print("Dataset loaded successfully")

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
    # return inputs.input_features
    return {"input_features": inputs.input_features}

# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
librispeech = librispeech.cast_column("audio", Audio(sampling_rate=16000))
# ted_lium = ted_lium.cast_column("audio", Audio(sampling_rate=16000))


# common_voice = common_voice.map(preprocess_function, batched=True, remove_columns=common_voice.column_names)
# librispeech = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names)
librispeech = librispeech.map(preprocess_function, batched=True, remove_columns=librispeech.column_names, cache_file_name="cached_librispeech")


# ted_lium = ted_lium.map(preprocess_function, batched=True, remove_columns=ted_lium.column_names)

# combined_dataset = concatenate_datasets([librispeech, ted_lium])
combined_dataset = concatenate_datasets([librispeech])


dummy_input = torch.randn(1, 80, 3000)
torch.onnx.export(model, dummy_input, "whisper_model.onnx", opset_version=11)

ov_model = mo.convert_model("whisper_model.onnx", model_name="whisper_model")
serialize(ov_model, "whisper_model.xml")

model = ie.read_model("whisper_model.xml")


COMPRESSION_MODE = nncf.parameters.CompressWeightsMode.INT4_SYM
MAX_DROP = 0.2
# We consider the following range of parameters: group_size - [64, 128], ratio - [0.5,...,1.0]
MIN_GROUP_SIZE = 64
MAX_GROUP_SIZE = 128
MIN_RATIO = 0.5
MAX_RATIO = 1.0
RATIO_STEP = 0.1

def compress_model(
    ov_model: ov.Model, nncf_dataset: nncf.Dataset, ratio: float, group_size: int, awq: bool
) -> ov.Model:
    """
    Compress the given OpenVINO model using NNCF weight compression.

    :param ov_model: The original OpenVINO model to be compressed.
    :param nncf_dataset: A representative dataset for the weight compression algorithm.
    :param ratio: The ratio between baseline and backup precisions
    :param group_size: Number of weights (e.g. 128) in the channel dimension
        that share quantization parameters (scale).
    :param awq: Indicates whether use AWQ weights correction.
    :return: The OpenVINO model with compressed weights.
    """
    optimized_ov_model = nncf.compress_weights(
        ov_model.clone(),  # we should clone the model because `compress_weights` works in place
        dataset=nncf_dataset,
        mode=COMPRESSION_MODE,
        ratio=ratio,
        group_size=group_size,
        awq=awq,
        sensitivity_metric=nncf.parameters.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    )
    return optimized_ov_model

config = {
    "compression": {
        "algorithms": [
            {
                "name": "DefaultQuantization",
                "params": {
                    "preset": "mixed",
                    "stat_subset_size": 300
                }
            }
        ]
    }
}

calibration_dataset = [{"input_features": item["input_features"]} for item in librispeech]

print("Starting quantization...")
quantized_model = ie.quantize_model(model, config, calibration_dataset)

serialize(quantized_model, "whisper_quantized.xml")


# class DatasetWrapper:
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def get_inference_data(self):
#         for item in self.dataset:
#             yield item['input_features']

#     def __len__(self):
#         return len(self.dataset)

# wrapped_dataset = DatasetWrapper(combined_dataset)


# quantizer = OVQuantizer.from_pretrained(model)
# quantized_model = quantizer.quantize(
#     save_directory="./quantized_whisper_model_small",
#     calibration_dataset=combined_dataset,
#     batch_size=8,
#     weights_only=False,
#     quantization_config={
#         "compression": {
#             "algorithms": [
#                 {
#                     "name": "DefaultQuantization",
#                     "params": {
#                         "preset": "mixed",
#                         "stat_subset_size": 150
#                     }
#                 }
#             ]
#         }
#     }
# )

# quantization_config = nncf.CompressedModelManager.create(
#     model,
#     compression_config={
#         "algorithm": "quantization",
#         "preset": "mixed",
#         "stat_subset_size": 150
#     }
# )


# print("Starting quantization...")
# model.model = nncf.compress_weights(
#     model.model, 
#     dataset=wrapped_dataset,
#     mode=nncf.CompressWeightsMode.INT8,
#     ratio=0.6,
# )
# quantization_config = {
#     "algorithm": "quantization",
#     "preset": "mixed",
#     "stat_subset_size": 150
# }

model.eval()
model.save_pretrained("./quantized_whisper_model_small")

print("Starting quantization...")
# quantized_model = quantize(
#     model,
#     calibration_dataset=combined_dataset,
#     compression_config=quantization_config,
#     batch_size=8  # You can adjust this based on your system's capabilities
# )
# quantized_model = quantize(
#     model,
#     nncf.convert(quantization_config),
#     dataset=combined_dataset,
#     batch_size=8  # You can adjust this based on your system's capabilities
# )

print("Quantization completed successfully")

# torch.save(quantized_model.state_dict(), "./quantized_whisper_model_small.pth")
# quantized_model.save("./quantized_whisper_model_small")
print("Quantized model saved")

ov_model = OVModelForCausalLM.from_pretrained(
    "./quantized_whisper_model_small",
    export=True,
    preprocessor=processor
)
# ov_model.load_state_dict(torch.load("./quantized_whisper_model_small.pth"))
print("Model loaded in OpenVINO format")

ie = Core()
compiled_model = ie.compile_model(ov_model.model, device_name=device)
print(f"Model compiled for {device}")

ov_model.save_pretrained("./ov_whisper_model")

compiled_model = ie.compile_model(ov_model.model, device_name=device)
print(f"Model compiled for {device}")

# Test the model
audio = librispeech[0]["audio"]["array"]
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

start_t = time.time()
output = ov_model.generate(input_features=input_features)
print("Elapsed time: ", time.time() - start_t)

output_text = processor.batch_decode(output, skip_special_tokens=True)
print(output_text)


print("Model quantized, optimized, and saved successfully!")