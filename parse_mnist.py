import os
import struct
import numpy as np
import cv2


def parse_mnist_images(filename):
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        rows = struct.unpack('>i', file_content[8:12])[0]
        cols = struct.unpack('>i', file_content[12:16])[0]
        # 'item_number * rows * cols' is the number of bytes
        images = struct.unpack(
            '>%dB' % (item_number * rows * cols), file_content[16:])
        images = np.uint8(np.array(images))
        # np.reshape: the dimension assigned by -1 will be computed according
        # to the first input (images) and other dimensions (rows, cols)
        images = np.reshape(images, [-1, rows, cols])
    return images


def parse_mnist_labels(filename):
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        # 'item_number' is the number of bytes
        labels = struct.unpack('>%dB' % item_number, file_content[8:])
        labels = np.array(labels)
    return labels


def make_one_hot_labels(labels):
    classes = np.unique(labels)
    assert len(classes) == classes.argmax() - classes.argmin() + 1
    labels_one_hot = (labels[:, None] == np.arange(10)).astype(np.int32)
    return labels_one_hot