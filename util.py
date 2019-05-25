import numpy as np
import pickle
import tensorflow as tf


def unpickle(my_file):
    """ Unpickle the data """
    with open(my_file, 'rb') as in_file:
        output_dict = pickle.load(in_file, encoding='bytes')
        return output_dict


def split_images_and_labels(data):
    """ Split the data into images and labels """
    return data[b'data'], np.array(data[b'fine_labels'])


def split_labels(labels):
    """ Split fine labels and coarse labels """
    return labels[b'fine_label_names'], labels[b'coarse_label_names']


def construct_superclass_mapping(fine_labels, super_labels):
    """ Build a mapping list from fine labels to superclass labels """
    pairs = set(zip(fine_labels, super_labels))
    class_mapping = [s_label for _, s_label in sorted(pairs, key=lambda x: x[0])]
    return class_mapping


def decode_binary_text(binary_text):
    """ Decode binary texts to regular texts """
    return [text.decode('ascii') for text in binary_text]


def map_class(fine_labels, mapping):
    """ Map 1-D array of fine class labels to their super class labels """
    return np.array([mapping[l] for l in fine_labels])


def map_text_labels(int_labels, text_label_mapping):
    """ Map integers to their text labels """
    return [text_label_mapping[l] for l in int_labels]


def format_data(data):
    """ Pre-process the data into the format that we want """
    data.shape = (-1, 3, 32, 32)
    return data.transpose((0, 2, 3, 1))


def split_train_and_validation(data, labels, split):
    """ Split the entire data into training and validation dataset """
    return data[:split], labels[:split], data[split:], labels[split:]


def get_random_batch(data, labels, batch_size):
    """ Select random batch from the data """
    order = np.random.choice(len(data), size=batch_size, replace=False)
    return data[order], labels[order]


def map_all_classes(fine_labels, mapping):
    """ Map all fine class labels to their corresponding super class labels """
    rows, cols = fine_labels.shape

    for r in range(rows):
        for c in range(cols):
            fine_labels[r][c] = mapping[fine_labels[r][c]]


def combine_ten_images(images):
    """ Combine ten images for display """
    if len(images) != 10:
        raise AttributeError("Must have more than one image to combine images.")

    row1 = np.hstack((images[0], images[1], images[2], images[3], images[4]))
    row2 = np.hstack((images[5], images[6], images[7], images[8], images[9]))

    return np.vstack((row1, row2))


def correct_in_top_5_super(labels, true_labels):
    """ Calculate the correctness of the top 5 labels """
    if len(labels) != len(true_labels):
        raise AttributeError("The first dimension must match.")

    num_of_labels = len(labels)
    correctness = [False] * num_of_labels
    for r in range(num_of_labels):
        if true_labels[r] in labels[r]:
            correctness[r] = True

    return correctness


def lenet_5(data):
    """ LeNet5 network structure """

    y_1 = tf.layers.conv2d(
        name='ConvLayer0',
        inputs=data,
        filters=6,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y_1_1 = tf.layers.conv2d(
        name='ConvLayer1',
        inputs=y_1,
        filters=12,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y_2 = tf.layers.max_pooling2d(
        name='MaxPoolingLayer1',
        inputs=y_1_1, pool_size=2, strides=2)

    y_3 = tf.layers.conv2d(
        name='ConvLayer3',
        inputs=y_2,
        filters=12,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y_3_1 = tf.layers.conv2d(
        name='ConvLayer4',
        inputs=y_3,
        filters=18,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y_4 = tf.layers.max_pooling2d(
        name='MaxPoolingLayer2',
        inputs=y_3_1, pool_size=2, strides=2)

    y_4_reshaped = tf.layers.flatten(y_4)   # reshape to fit the dense layer

    y_5 = tf.layers.dense(
        name='DenseLayer1',
        inputs=y_4_reshaped,
        units=120,
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y_6 = tf.layers.dense(
        name='DenseLayer2',
        inputs=y_5,
        units=84,
        activation=tf.nn.relu,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    y = tf.layers.dense(
        name='DenseLayer3',
        inputs=y_6,
        units=100,
        activation=None,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer())

    return y
