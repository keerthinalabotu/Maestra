from config import TEST_FILE, SAMPLE_AUDIO, SAMPLE_AUDIO_MY_VOICE, SAMPLE_AUDIO_VOICE_2
# from input_processor import ProcessInput
# from input_processor_copy import ProcessInput
from input_processor_chunking_audio import ProcessInput
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from clients.llama_client import LlamaClient
from clients.phi3_client_w_openvino import Phi3Client
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from openvino.runtime import Core
from huggingface_hub import hf_hub_download
import os
import numpy as np
from vector_store import VectorStore
# import rag_testing


model_path = "./clients/whisper_model_small"
processor_path = "./clients/whisper_processor_small"
  
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
whisper_processor = WhisperProcessor.from_pretrained(processor_path)

processor = ProcessInput(whisper_model, whisper_processor)

text = processor.process_pdf(TEST_FILE)
speech_text = processor.process_speech(SAMPLE_AUDIO_VOICE_2)
# print(processor.process_pdf(TEST_FILE))
print(speech_text)

from transformers import AutoModel, AutoTokenizer

phi3_client = Phi3Client()
phi3_client.load_model()

print("model loaded")
# model_name = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("tokenizer done")
# model = OVModelForCausalLM.from_pretrained(model_name)
# print("tokenizer model loaded")


# model_name = "microsoft/Phi-3-mini-4k-instruct"


# model = AutoModelForCausalLM.from_pretrained(model_name)
 
# inputs = tokenizer(text, return_tensors="pt")

inputs = phi3_client.tokenizer(text, return_tensors="pt")

input_ids = inputs['input_ids']
# input_ids_np = input_ids.numpy()

# input_ids_np = input_ids.numpy().astype(np.int32)
# input_ids_np = input_ids_np.reshape((1, -1))  # Adjust to match the expected shape


print(f"Input IDs: {input_ids}")
# print(f"Input IDs: {input_ids_np}")
print(f"Input IDs shape: {input_ids.shape}")

phi3_client.model.to("gpu")
phi3_client.model.compile()
# model.to("gpu")
# model.compile()
# print(f"Input IDs shape: {input_ids_np.shape}")

# model_file = hf_hub_download(repo_id=model_name, filename="openvino_model.xml")
# weights_file = hf_hub_download(repo_id=model_name, filename="openvino_model.bin")

# model_dir = os.path.dirname(model_file)

ie = Core()

# model = ie.read_model(model_file)
# compiled_model = ie.compile_model(model, "CPU") 


print("model loaded")
# This will download and cache the model
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("model cached")

# Print the cache directory
# print(f"Model cached at: {model.config.cache_dir}")

# model_path = model.config.cache_dir

# phi3_client = LlamaClient(model, tokenizer)

# # llama_client = LlamaClient(model_path="OpenVINO/open_llama_3b_v2-int8-ov")
# # llm = llama_client.load_model()

# response=phi3_client.generate_response(text, 3000)
# print("---------------------------the llama client response")
# print(response)

# phi3_client = Phi3Client(model, tokenizer)

# llama_client = LlamaClient(model_path="OpenVINO/open_llama_3b_v2-int8-ov")
# llm = llama_client.load_model()
print("-------------------------------the phi client response")
# response=phi3_client.generate_response(text, 3000)
# print(response)

overview_of_info= phi3_client.initial_overview(text)

print(overview_of_info)

storage = VectorStore()

storage.test_chroma_storage()
storage.add_pdf_text_file_to_collection("Civil War Lincoln", text, "CivilWar", "History")

print("list collection")
print(storage.list_collections())

print("get collection")
print(storage.get_collection_info("CivilWar"))
print(storage.get_collection_info("CivilWar-History"))

storage.inspect_collection("CivilWar-History")

print("searching in context")
print(storage.search_context("What was a turning point of the Civil War?", topic="CivilWar", field="History", n_results=1))
# relevant_info = rag_testing.get_relevant_info("the civil war was during a time when the south was slavery centered.")

# print(relevant_info)