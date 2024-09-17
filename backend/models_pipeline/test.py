from config import TEST_FILE
from input_processor import ProcessInput
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from clients.llama_client import LlamaClient
from transformers import AutoModel, AutoTokenizer
from openvino.runtime import Core
from huggingface_hub import hf_hub_download
import os
import numpy as np


model_path = "./clients/whisper_model_small"
processor_path = "./clients/whisper_processor_small"
  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
whisper_processor = WhisperProcessor.from_pretrained(processor_path)

processor = ProcessInput(whisper_model, whisper_processor)

text = processor.process_pdf(TEST_FILE)
print(processor.process_pdf(TEST_FILE))

from transformers import AutoModel, AutoTokenizer

print("model loading")
model_name = "OpenVINO/open_llama_3b_v2-int8-ov"
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids']
input_ids_np = input_ids.numpy()

input_ids_np = input_ids.numpy().astype(np.int32)
input_ids_np = input_ids_np.reshape((1, -1))  # Adjust to match the expected shape


print(f"Input IDs: {input_ids}")
print(f"Input IDs: {input_ids_np}")
print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs shape: {input_ids_np.shape}")

model_file = hf_hub_download(repo_id=model_name, filename="openvino_model.xml")
weights_file = hf_hub_download(repo_id=model_name, filename="openvino_model.bin")

model_dir = os.path.dirname(model_file)

ie = Core()

model = ie.read_model(model_file)
compiled_model = ie.compile_model(model, "CPU") 


print("model loaded")
# This will download and cache the model
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("model cached")

# Print the cache directory
# print(f"Model cached at: {model.config.cache_dir}")

# model_path = model.config.cache_dir

llama_client = LlamaClient(compiled_model, tokenizer)

# llama_client = LlamaClient(model_path="OpenVINO/open_llama_3b_v2-int8-ov")
# llm = llama_client.load_model()

llama_client.generate_response(text, 3000)