Notes files to document the process.

### SSD Object detection does not work in TF 1.3

The SSD was exported using a Tensorflow version higher than 1.3, so testing on the Udacity VM the following error appears:

```bash
Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.
```
The model needs to be exported from a TF 1.3 installation.

The [object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) models were installed.

But it does no work by default, the following tweaks were applied:

* Commented the 'tf.app.flags.mark_flag_as_required' lines (3 lines of code) from [object_detection/export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py#L123)
* Commented the 'reserved 6;' line from [object_detection/protos/ssd.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/ssd.proto#L87)

Assuming the repository is available in 'Our_CarND_Capstone', execute the following

```bash
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ~/Our_CarND_Capstone/ros/ssd_bags_20K/pipeline.config \
    --trained_checkpoint_prefix ~/Our_CarND_Capstone/ros/ssd_bags_20K/model.ckpt \
    --output_directory ~/Our_CarND_Capstone/ros/ssd_bags_20K_tf1.3
```

#### Discarded solutions

* [pb_model_executor.py](https://gist.github.com/peci1/80cf0dd79986db83b4c99d0714ddf2ff) 
* [Object detection works on Linux but not Mac](https://github.com/tensorflow/tensorflow/issues/14884) 

