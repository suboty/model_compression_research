# model_compression_research

ViT for example. 
Testing was carried out on a one RTX 8000 video card and on a CPU with 32 cores. 
There are 150 photos with cats and gogs

| Models                    | Number of photos in one second | General time | Model Size (Mb) |
|:--------------------------|:------------------------------:|:------------:|:---------------:|
| original ViT              |             0.2541             |   590.319    |     327.302     |
| converted to ONNX ViT     |             10.594             |    14.159    |     327.552     |
| converted to TensorRT ViT |             27.243             |     5.50     |     327.302     |
| converted to OpenVINO ViT |             7.9526             |    18.86     |     163.65      |