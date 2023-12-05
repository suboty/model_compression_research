import torch
import openvino
import numpy as np
from scipy.special import softmax
from openvino.tools import mo
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class OpenVINOViTModel:
    def __init__(self, path_to_model):
        self.extractor = AutoFeatureExtractor.from_pretrained(path_to_model)
        self.model = AutoModelForImageClassification.from_pretrained(path_to_model)

        self.compiled_openvino_model = None
        self.convert_model()

    def get_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024 ** 2)

    def convert_model(self):
        openvino_model = mo.convert_model(self.model,
                                       example_input=torch.randn(1, 3, 224, 224))

        core = openvino.Core()
        self.compiled_openvino_model = core.compile_model(openvino_model, "CPU")

    @staticmethod
    def postprocess_result(result, top_n):
        scores = softmax(result, -1)[0]
        top_labels = np.argsort(scores)[-top_n:][::-1]
        top_scores = scores[top_labels]
        return top_labels, top_scores

    def predict(self, _model, img):
        inputs = self.extractor(img, return_tensors="pt")["pixel_values"]
        result = self.compiled_openvino_model(inputs)[0]
        predict, score = self.postprocess_result(result, top_n=1)

        return predict
