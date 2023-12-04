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
        self.images_list = [x for x in os.listdir(path_to_images) if x != '.gitkeep']
        self.models = []

    @staticmethod
    def _count_accuracy(target_list, predict_list):
        if len(target_list) != len(predict_list):
            raise AttributeError(f'Error with equality lengths. '
                                 f'Target len: {len(target_list)}.'
                                 f'Predict len: {len(predict_list)}')

    def run(self):
        start_time = time.time()

        target_list = []
        predict_list = []

        for model in self.models:

            logger.info(f'Model {model[0]}')

            try:

                for image_name in self.images_list:

                    image = Image.open(Path(self.path_to_images, image_name), mode='r', formats=None)
                    predict = model[1].predict(_model=model[1], img=image)
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

                logger.info(f"--- Time for model processing = {end_time - start_time} seconds")
                logger.info(f"--- Speed of model processing is "
                            f"{len(self.images_list) / (end_time - start_time)} images per second")

                logger.info(f'--- Model {model[0]} Accuracy is: '
                            f'{self._count_accuracy(target_list=target_list, predict_list=predict_list)}')

            except Exception as e:
                logger.critical(f'--- Error with model {model[0]}. Error: {e}')


if __name__ == '__main__':
    test_compression_run = TestRun(path_to_images=Path('data'))

    original_model = ViTModel(path_to_model=Path('model'))
    openvino_model = OpenVINOViTModel(path_to_model=Path('model'))
    onnx_model = ONNXViTModel(path_to_model=Path('model'))
    tensorrt_model = TensorRTViTModel(path_to_model=Path('model'))

    test_compression_run.models.append(('original_model', original_model))
    test_compression_run.models.append(('openvino_model', openvino_model))
    test_compression_run.models.append(('onnx_model', onnx_model))
    test_compression_run.models.append(('tensorrt_model', tensorrt_model))

    test_compression_run.run()
