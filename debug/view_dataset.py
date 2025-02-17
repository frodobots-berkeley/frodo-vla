import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
from PIL import Image
import glob



builder = tfds.builder("cf_v2_dataset_128", data_dir="gs://vlm-guidance-data/cleaned")
paths = tf.io.gfile.glob(f"{builder.data_path}/*.tfrecord*")
features = builder.info.features

path = paths[0]
dataset = tf.data.TFRecordDataset([path]).map(features.deserialize_example)

for example in dataset:

    breakpoint()
