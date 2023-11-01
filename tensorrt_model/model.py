import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class TensorRTViTModel:
    def __init__(self, path_to_model):
        self.extractor = AutoFeatureExtractor.from_pretrained(path_to_model)
        self.model = AutoModelForImageClassification.from_pretrained(path_to_model)

        self.tensorrt_model = None

    def predict(self, img):
        model_id2label = {0: "cats", 1: "dogs"}
        output = self.tensorrt_model(img.cuda())

        predicted_label = output.logits[0].argmax(-1).item()

        return model_id2label[predicted_label]

    def _get_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024 ** 2)

    def model_convert(self):
        model = self.model.eval().to("cuda")

        enabled_precisions = {torch.float}
        debug = True
        workspace_size = 20 << 30
        min_block_size = 7
        torch_executed_ops = {}
        compilation_kwargs = {
            "enabled_precisions": enabled_precisions,
            "debug": debug,
            "workspace_size": workspace_size,
            "min_block_size": min_block_size,
            "torch_executed_ops": torch_executed_ops,
        }

        self.tensorrt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            options=compilation_kwargs,
        )
