import os
import time
import traceback
from pathlib import Path

from PIL import Image

from logger import logger
from original_model.model import ViTModel
from openvino_model.model import OpenVINOViTModel
from onnx_model.model import ONNXViTModel
from tensorrt_model.model import TensorRTViTModel

import warnings
warnings.simplefilter(action='ignore')


class TestRun:
    def __init__(self,
                 path_to_images: Path):
        self.path_to_images = path_to_images

        self.datasets = {}

        for dataset in os.listdir(path_to_images):
            self.datasets[dataset] = [x for x in
                                      os.listdir(os.path.join(os.getcwd(),
                                                              path_to_images,
                                                              dataset)) if x != '.gitkeep']
        self.models = []

    @staticmethod
    def _count_accuracy(target_list, predict_list):
        if len(target_list) != len(predict_list):
            raise AttributeError(f'Error with equality lengths. '
                                 f'Target len: {len(target_list)}.'
                                 f'Predict len: {len(predict_list)}')

        accuracy = (sum([x == y for x, y in zip(predict_list, target_list)]))/len(target_list)
        return accuracy

    def run(self):
        start_time = time.time()

        target_list = []
        predict_list = []

        for model in self.models:

            logger.info(f'Model {model[0]}')
            logger.info(f'--- Model size is {model[1].get_size()}')

            try:
                for dataset_name in self.datasets.keys():

                    logger.info(f'--- Dataset {dataset_name}')

                    if len(self.datasets[dataset_name]) == 0:
                        continue

                    for image_name in self.datasets[dataset_name]:

                        image = Image.open(Path(self.path_to_images,
                                                dataset_name,
                                                image_name), mode='r', formats=None)
                        predict = model[1].predict(_model=model[1], img=image)
                        target = image_name[:image_name.find(".")]

                        target_list.append(1 if target == 'dog' else 0)
                        predict_list.append(1 if predict == 'dogs' else 0)

                    end_time = time.time()

                    logger.info(f"------ Time for model processing = {end_time - start_time} seconds")
                    logger.info(f"------ Speed of model processing is "
                                f"{len(self.datasets[dataset_name]) / (end_time - start_time)} images per second")

                    logger.info(f'------ Model {model[0]} Accuracy is: '
                                f'{self._count_accuracy(target_list=target_list, predict_list=predict_list)}')

            except Exception as e:
                logger.critical(f'--- Error with model {model[0]}. Error: {traceback.format_exc()}')


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
