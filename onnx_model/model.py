import os
from pathlib import Path

import onnxruntime
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class ONNXViTModel:
    def __init__(self, path_to_model):
        self.extractor = AutoFeatureExtractor.from_pretrained(path_to_model)
        self.model = AutoModelForImageClassification.from_pretrained(path_to_model)

        self.path_to_onnx_model = "vit.onnx"
        self.ort_session = None

    def predict(self, img):

        model_id2label = {0: "cats", 1: "dogs"}
        ort_inputs = {self.ort_session.get_inputs()[0].name: img['pixel_values'].numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)[0]

        predicted_label = ort_outs.argmax(-1).item()

        return model_id2label[predicted_label]

    def _get_size(self):
        return os.path.getsize(self.path_to_onnx_model)/(1024**2)

    def convert_model(self):
        image = Image.open(Path('onnx_model', 'dummy_image.jpg'), mode='r', formats=None)
        inputs_onnx = self.extractor(image, return_tensors="pt")

        torch.onnx.export(self.model,
                          {'pixel_values': inputs_onnx['pixel_values']},
                          self.path_to_onnx_model,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['pixel_values'],
                          output_names=['output'],
                          dynamic_axes={'pixel_values': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        self.ort_session = onnxruntime.InferenceSession(self.path_to_onnx_model)
