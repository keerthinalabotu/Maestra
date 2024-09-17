# from openvino.runtime import Core
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import numpy as np
import json
import librosa
import logging
import os
import torch
import time
from optimum.intel import OVModelForCausalLM

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class Phi3Client:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.wrong_questions = []
        self.current_context = ""

    def load_model(self):
        logger.info("Starting to load Phi-3 model")

        # model_id = "microsoft/phi-3"

        logger.info("Loading tokenizer")
        model_name = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")

        self.model = OVModelForCausalLM.from_pretrained(model_name)

        logger.info("Loading model")
        # self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
        logger.info("Model loaded successfully")

        self.model.to('gpu')

        return self
    
    # def generate_response_openvino (self, prompt, max_length = 1000):


    def generate_response(self, prompt, max_length=1000):
        print(f"Padding Token: {self.tokenizer.pad_token}")
        print(f"EOS Token: {self.tokenizer.eos_token}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        print("Initial input_ids shape:", input_ids.shape)

        start_time = time.time()
        with torch.no_grad():
            print("helloooo?!")

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens = max_length,
                num_return_sequences=1,
                # do_sample=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.95,
                use_cache=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        end_time = time.time()
        logger.info(f"Time taken for generation: {end_time - start_time:.2f} seconds")

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def initial_overview(self, input_text):
        prompt = f"Provide an initial overview of the following text in 5 sentences, followed by a relevant question:\n\n{input_text}\n\nOverview:"
        response = self.generate_response(prompt)
        self.current_context = response
        return response

    def ask_follow_up(self):
        prompt = f"{self.current_context}\n\nAsk a follow-up question based on the above context:"
        response = self.generate_response(prompt)
        self.current_context += f"\n\nFollow-up question: {response}"
        return response

    def verify_answer(self, question, student_answer):
        prompt = f"{self.current_context}\n\nQuestion: {question}\nStudent's answer: {student_answer}\n\nIs the student's answer correct? Respond with 'Correct' or 'Incorrect' and provide a brief explanation:"
        response = self.generate_response(prompt)
        self.current_context += f"\n\n{response}"
        return response

    def next_question(self, is_correct):
        if not is_correct:
            self.wrong_questions.append(self.current_context.split("\n")[-1])
        
        if is_correct:
            prompt = f"{self.current_context}\n\nProvide a new, related question based on the above context:"
        else:
            prompt = f"{self.current_context}\n\nThe previous answer was incorrect. Provide a new, different question based on the above context:"
        
        response = self.generate_response(prompt)
        self.current_context += f"\n\nNew question: {response}"
        return response

    def get_wrong_question(self):
        if self.wrong_questions:
            return self.wrong_questions.pop(0)
        return None

# class LlamaClient:
#     def __init__(self, compiled_model, tokenizer):
#         # self.ie=Core()
#         self.model = compiled_model
#         self.tokenizer = tokenizer
#         # self.output_layer = self.compiled_model.output(0)
#         self.wrong_questions = []
#         self.current_context = ""
#         # self.model = self.ie.read_model(model_path)
#         # self.compiled_model = self.ie.compile_model(self.model, "CPU")  # Use "GPU" if available
#         # self.output_layer = self.compiled_model.output(0)
#         # self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
#         # self.wrong_questions = []
#         # self.current_context = "" 

#     def load_model(self): 
#         logger.info("Starting to load Llama model")


#         logger.info("Loading tokenizer")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.getenv("HUGGING_FACE_TOKEN"))
#         logger.info("Tokenizer loaded successfully")

#         logger.info("Loading model")
        
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_id, token=os.getenv("HUGGING_FACE_TOKEN"))

#         logger.info("Model loaded successfully")

#         model = self.ie.read_model(self.model_id)
#         self.compiled_model = self.ie.compile_model(model, "CPU")  # Use "GPU" if available
#         logger.info("OpenVINO model loaded and compiled successfully")


#         # logger.info("Creating pipeline")
#         # self.pipe = pipeline(
#         #     "text-generation",
#         #     model = self.model,
#         #     tokenizer = self.tokenizer,
#         #     max_length =10000,
#         #     temperature = 0.7,
#         #     top_p = 0.95,
#         #     repetition_penalty = 1.15
#         # )

#         # logger.info("Pipeline created successfully")

#         # logger.info("Creating HuggingFacePipeline")
#         # self.llm = HuggingFacePipeline(pipeline = self.pipe)
#         # logger.info("HuggingFacePipeline created successfully")

#         return self
    
#     def generate_response(self, prompt, max_length):
#         # input_ids = self.tokenizer.encode(prompt, return_tensors = "np")

#         # print("Initial input_ids shape:", input_ids.shape)
#         print(f"Padding Token: {self.tokenizer.pad_token}")
#         print(f"EOS Token: {self.tokenizer.eos_token}")
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
    
#         inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']
#         print("Initial input_ids shape:", input_ids.shape)

#         start_time = time.time()
#         # Generate response
#         with torch.no_grad():
#             print("helloooo?!")
#             outputs = self.model.generate(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 max_length=max_length,
#                 num_return_sequences=1,
#                 no_repeat_ngram_size=2,  # Optional: Prevent repeating n-grams
#                 temperature=0.7,         # Optional: Control randomness
#                 top_p=0.95,              # Optional: Control diversity
#                 pad_token_id=self.tokenizer.pad_token_id
#             )
#         end_time = time.time()
#         print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
#         print("Generating response!")
#         # Decode and return the response
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("Response generated!")
#         return response

#         # for i in range(max_length):
#         #     # outputs = self.compiled_model([input_ids])
#         #     try:
#         #         print("Input before model inference:", input_ids)
#         #         outputs = self.compiled_model([input_ids])
#         #         print("Outputs shape:", [output.shape for output in outputs])
#         #     except Exception as e:
#         #         print(f"Error during model inference: {e}")
#         #         break
#         #     next_token_logits = outputs[self.output_layer]
#         #     next_token = np.argmax(next_token_logits[:, -1, :])
#         #     input_ids = np.concatenate([input_ids, [[next_token]]], axis=-1)
            
#         #     if next_token == self.tokenizer.eos_token_id:
#         #         break
        
#         # return self.tokenizer.decode(input_ids[0])
    

#     def cache_info(info, cache_file="info_cache.json"):
#         with open(cache_file, "a") as f:
#             json.dump(info, f)
#             f.write("\n")

#     def read_cache(cache_file="info_cache.json"):
#         with open(cache_file, "r") as f:
#             return [json.loads(line) for line in f]

#     def initial_overview(self, input_text):
#         prompt = f"Provide an initial overview of the following text in 5 sentences, followed by a relevant question:\n\n{input_text}\n\nOverview:"
#         response = self.generate_response(prompt)
#         self.current_context = response
#         return response

#     def ask_follow_up(self):
#         prompt = f"{self.current_context}\n\nAsk a follow-up question based on the above context:"
#         response = self.generate_response(prompt)
#         self.current_context += f"\n\nFollow-up question: {response}"
#         return response

#     def verify_answer(self, question, student_answer):
#         prompt = f"{self.current_context}\n\nQuestion: {question}\nStudent's answer: {student_answer}\n\nIs the student's answer correct? Respond with 'Correct' or 'Incorrect' and provide a brief explanation:"
#         response = self.generate_response(prompt)
#         self.current_context += f"\n\n{response}"
#         return response

#     def next_question(self, is_correct):
#         if not is_correct:
#             self.wrong_questions.append(self.current_context.split("\n")[-1])
        
#         if is_correct:
#             prompt = f"{self.current_context}\n\nProvide a new, related question based on the above context:"
#         else:
#             prompt = f"{self.current_context}\n\nThe previous answer was incorrect. Provide a new, different question based on the above context:"
        
#         response = self.generate_response(prompt)
#         self.current_context += f"\n\nNew question: {response}"
#         return response

#     def get_wrong_question(self):
#         if self.wrong_questions:
#             return self.wrong_questions.pop(0)
#         return None


