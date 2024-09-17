from optimum.intel import INCModelForSequenceClassification

model_id = "Intel/whisper.cpp-openvino-models"
model = INCModelForSequenceClassification.from_pretrained(model_id)