import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class ViTModel:
    def __init__(self, path_to_model):
        self.extractor = AutoFeatureExtractor.from_pretrained(path_to_model)
        self.model = AutoModelForImageClassification.from_pretrained(path_to_model)

    def predict(self, _model, img):
        inputs = _model.extractor(img, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_label = logits.argmax(-1).item()

        return self.model.config.id2label[predicted_label]

    def _get_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024 ** 2)
