
import tensorflow as tf
from PIL import Image
import os
import dataset_util
import json
import hashlib
import argparse
from tqdm import tqdm
from random import shuffle


def generate_class_num(points):
    output = []
    with open(trainable_classes_file, 'r') as file:
        trainable_classes = file.read().split('\n')
    for point in tqdm(points):
        for anno in point['annotations']:
            anno['class_num'] = trainable_classes.index(anno['label'])+1
            output.append(anno)
    return output


# Construct a record for each image.
# If we can't load the image file properly lets skip it
def group_to_tf_record(point, image_directory):
    format_png = b'png'
    format_jpg = b'jpeg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_nums = []
    class_ids = []
    # changed point[0] to point as is just one point
    image_id = point['id']

    if image_id.startswith('frame'):
        filename = os.path.join(image_directory, image_id + '.png')
        format = format_png
    else:
        filename = os.path.join(image_directory, image_id + '.jpg') #.decode()
        format = format_jpg

    try:
        image = Image.open(filename)
        width, height = image.size
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_image = bytes(fid.read())
    except:
        return None
    key = hashlib.sha256(encoded_image).hexdigest()
    for anno in point['annotations']:
        xmins.append(float(anno['x0']))
        xmaxs.append(float(anno['x1']))
        ymins.append(float(anno['y0']))
        ymaxs.append(float(anno['y1']))
        class_nums.append(anno['class_num'])
        class_ids.append(bytes(anno['label'].encode('utf8')))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/filename': dataset_util.bytes_feature(bytes(filename.encode('utf8'))),
        'image/source_id': dataset_util.bytes_feature(bytes(image_id.encode('utf8'))),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_ids),
        'image/object/class/label': dataset_util.int64_list_feature(class_nums)
    }))
    return tf_example


def load_points(file_path):
    with open(file_path, 'rb') as f:
        points = json.load(f)
    return points

parser = argparse.ArgumentParser()
parser.add_argument('--points_path', dest='points_file_path', required=True)
parser.add_argument('--train_record_save_path', dest='train_record_save_path', required=True)
parser.add_argument('--eval_record_save_path', dest='eval_record_save_path', required=True)
parser.add_argument('--trainable_classes_path', dest='trainable_classes_path', required=True)
parser.add_argument('--saved_images_directory', dest='saved_images_directory', required=True)
if __name__ == "__main__":
    args = parser.parse_args()
    trainable_classes_file = args.trainable_classes_path
    train_record_save_path = args.train_record_save_path
    eval_record_save_path = args.eval_record_save_path
    points_file = args.points_file_path
    saved_images_directory = args.saved_images_directory

    points = load_points(points_file)
    
    shuffle(points) # Randomize

    split = int(len(points) * 0.8)  # get 20% of points for eval

    train =  points[0:split]
    evaluation = points[split:]


    with_class_num = generate_class_num(points)
    writer = tf.python_io.TFRecordWriter(train_record_save_path)
    for point in tqdm(train, desc="writing to training file"):
        record = group_to_tf_record(point, saved_images_directory)
        if record:
            serialized = record.SerializeToString()
            writer.write(serialized)
    writer.close()

    writer = tf.python_io.TFRecordWriter(eval_record_save_path)
    for point in tqdm(evaluation, desc="writing to evaluation file"):
        record = group_to_tf_record(point, saved_images_directory)
        if record:
            serialized = record.SerializeToString()
            writer.write(serialized)
    writer.close()

    print("All Done")


