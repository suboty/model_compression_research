import os
import time
from pathlib import Path

from PIL import Image

from logger import logger
from original_model.model import ViTModel
from openvino_model.model import OpenVINOViTModel
from onnx_model.model import ONNXViTModel
from tensorrt_model.model import TensorRTViTModel


class TestRun:
    def __init__(self,
                 path_to_images: Path):
        self.path_to_images = path_to_images
        self.images_list = os.listdir(path_to_images)
        self.models = []

    def run(self):
        start_time = time.time()

        target_list = []
        predict_list = []

        for model in self.models:

            for image_name in self.images_list:

                image = Image.open(Path(self.path_to_images, image_name), mode='r', formats=None)

                inputs = model.extractor(image, return_tensors="pt")
                predict = model.predict(inputs)
                target = image_name[:image_name.find(".")]

                if target == "dog":
                    label = 1
                else:
                    label = 0

                target_list.append(label)

                if predict == "dogs":
                    pr = 1
                else:
                    pr = 0

                predict_list.append(pr)

            end_time = time.time()

            logger.info("Время обработки изображений исходной модели= ", end_time - start_time, " секунд")
            logger.info("Скорость обработки изображений у исходной модели составила  ",
                        len(self.images_list) / (end_time - start_time), " изображений в секунду")


if __name__ == '__main__':
    test_compression_run = TestRun(path_to_images=Path('data'))

    original_model = ViTModel(path_to_model=Path('model'))
    openvino_model = OpenVINOViTModel(path_to_model=Path('model'))
    onnx_model = ONNXViTModel(path_to_model=Path('model'))
    tensorrt_model = TensorRTViTModel(path_to_model=Path('model'))

    test_compression_run.models.append(original_model)
    test_compression_run.models.append(openvino_model)
    test_compression_run.models.append(onnx_model)
    test_compression_run.models.append(tensorrt_model)

    test_compression_run.run()
