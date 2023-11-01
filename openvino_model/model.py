import torch
import openvino
from numpy import np
from scipy.special import softmax
from openvino.tools.mo import convert_model
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class OpenVINOViTModel:
    def __init__(self, path_to_model):
        self.extractor = AutoFeatureExtractor.from_pretrained(path_to_model)
        self.model = AutoModelForImageClassification.from_pretrained(path_to_model)

        self.compiled_openvino_model = None

    def _get_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024 ** 2)

    def convert_model(self):
        openvino_model = convert_model(self.model,
                                       example_input=torch.randn(1, 3, 224, 224))

        core = openvino.Core()
        self.compiled_openvino_model = core.compile_model(openvino_model, "CPU")

    @staticmethod
    def postprocess_result(result, top_n):
        softmaxed_scores = softmax(result, -1)[0]
        topk_labels = np.argsort(softmaxed_scores)[-top_n:][::-1]
        topk_scores = softmaxed_scores[topk_labels]
        return topk_labels, topk_scores

    def predict(self, img):
        inputs = self.extractor(img, return_tensors="pt")["pixel_values"]
        result = self.compiled_openvino_model(inputs)[0]
        predict, score = self.postprocess_result(result, top_n=1)

        return predict
