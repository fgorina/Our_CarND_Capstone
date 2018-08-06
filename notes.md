Notes files to document the Training process.

### Retraining SSD Object detection to classify 3 classes

#### Generate Images from simulation and bag files.

First we saved several bag files from the simulation. Then with the provided "real world" bags we extracted images using the [bag_to_images.py](./Preparation/bag_to_images.py) file.

We created a script to automatically [label](./Preparation/label_files.py) the images.

#### Generating CSV files from Images

The goal is to label the images and generate a csv text file. 

Initially we planned to use one of the object detection models which have good detection accuracy directly in the control loop but unfortunately the ones which provided good accuracy were too slow, so we decided a two part approach:

- Lookup the traffic light boxes with an accurate model. We used the __faster_rcnn_inception_v2_coco_2018_01_28__ from the object model zoo. It gave good traffic light box detection but was too slow to use in the ROS control loop.

- Retrain an SSD which will be much faster although probab ly less accurate.

To ompute the __color__ of the light we had a problem with the __real__ images. Sometimes (near always) the intensity of the active light was enough to saturate the camera that just gave __white__. So we could not use the __color__ of the pixels. We decided to use its position into the traffic light box detected by the f__aster_rcnn_inception_v2_coco_2018_01_28__. So red lights where labeled when the intensity of the top third was the maximum, yellow ones when highest intensity was in the middle and green ones when at the bottom.

- To compute the intensity we used just the average of the center pixels of every third of the returned box. 

```python
def get_vertical_color(img):

    h = img.shape[0]
    w = img.shape[1]

    dl = int(h/3)
    dx0 = int(h/9)
    dx1 = 2*dx0
    #print("dl : ", dl)


    red = np.sum(bimg[0:dl-1, dx0:dx1, :])
    orange = np.sum(bimg[dl:(2*dl)-1, dx0:dx1, :])
    green = np.sum(bimg[(dl*2):(dl*3)-1, dx0:dx1, :])


    if green > orange and green > red:
        color = "Green"

    elif orange > red and orange > green:
        color = "Orange"

    elif red > orange and red > green:
        color = "Red"

    else:
        color = "Unknown"
		
    return color

```

- Use all these automatically generated labels to retrain an SSD based in the __ssd_mobilenet_v2_coco_2018_03_29__ also from the model zoo. We used about 3600 images and 20000 iterations on an AWS machine.

We got much of the information to use the Google Object Detection libraries from the Algorithmia [Deep Dive into Object Detection with Open Images, using Tensorflow](https://blog.algorithmia.com/deep-dive-into-object-detection-with-open-images-using-tensorflow/).


Using the [format_annotations.py](./Preparation/format_annotations.py) file we create a json file with the  annotations and the image urls.


#### Generating TFRecords

TensorFlow object detection API doesn't take csv files as an input, but it needs record files to train the model.

The [record_maker.py](./Preparation/record_maker.py) file combines annotations and image files in a big tensor record array which is the input to the train algorithm.

As a plus the [record_maker.py](./Preparation/record_maker.py) file splits the set in two with 80% of the records for training and the other with 20% for evaluation purposes.

#### Training the Model

First, we decided which pre-trained model to be used. The selection was [ssd_mobilenet_v2_coco_2018_03_29](https://github.com/tensorflow/models/blob/master/research/object_detection/models/ssd_mobilenet_v2_feature_extractor.py).

Then, we downloaded the config file for the same model and changed the configuration.

Created a new label_map.pbtxt file containing the definition for the traffic light class we want to detect:

```
item {
id: 1
name: '/m/015qff'
}```

The .config file changed accordingly

```
...
num_classes: 3
...
batch_size: 18
PATH_TO_BE_CONFIGURED # location of files and resources
```

Once the detection is done, we use the box and our own algorithm for labelling the colors: UNKNOWN, RED, ORANGE, GREEN.

The train process:

```
python train.py --logtostderr \
       --train_dir=training/ \
 --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
```

#### Export the inference graph

To test the model and check it works we need to export the inference graph. An example command is:

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=training/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix=training/model.ckpt-66454 \
    --output_directory=new_inference_graph
```

Notice 66454 value will change on your execution.


### SSD Object detection does not work in TF 1.3

The SSD was exported using a Tensorflow version higher than 1.3, so testing on the Udacity VM the following error appears:

```bash
Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.
```
The model needs to be exported to a TF 1.3 installation.

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

