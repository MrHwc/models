# 要点  
### 1、多GPU训练
使用object_de_detection进行模型训练时，可通过设置[RunConfig](https://github.com/tensorflow/models/issues/5421#issuecomment-497372032)进行多GPU训练  

### 2、限制Slim模块GPU显存  
在使用Slim模块中的`train_image_classifier.py`训练NAS时，Slim默认使用全部GPU及其显存，需要对其进行限制。可将`slim.learning.train`中的`session_config`设置为`config`，`config`设置如下：  
```
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
config.gpu_options.allow_growth = True
```
