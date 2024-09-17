from openvino.runtime import Core
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVQuantizer, OVModelForCausalLM
from optimum.intel.optimization import ov_optimize_model
import torch

from openvino.runtime import Core

core = Core()
devices = core.available_devices
print(f"Available devices: {devices}")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

quantizer = OVQuantizer.from_pretrained(model)
quantized_model = quantizer.quantize(
    save_directory="./quantized_model",
    weights_only=False,  
    quantization_connfig={
        "compression": {
            "algorithms": [
                {
                    "name": "DefaultQuantization",
                    "params": {
                        "preset": "mixed",
                        "stat_subset_size": 100
                    }
                }
            ]
        }
    }
)


core = Core()
devices = core.available_devices
print(f"Available devices: {devices}")

ov_model = ov_optimize_model(
    quantized_model, 
    optimization_config={
        "compress_to_fp16": True,
        "enable_transformations": True,
        "device_type": "GPU"  
    }
)
ov_model.save_pretrained("./ov_quantized_model_gpu_npu")

tokenizer.save_pretrained("./ov_quantized_model")

print("Model quantized and saved successfully!")