import numpy as np
from parse_mnist import *
import matplotlib.pyplot as plt


def uniform_random(shape, min_limit, max_limit):
    return (max_limit - min_limit) * np.random.random(shape) + min_limit


def shuffle_data(images, labels_one_hot):
    images_shuffle = np.zeros_like(images)
    labels_shuffle = np.zeros_like(labels_one_hot)
    idx_shuffle = np.arange(images.shape[0])
    np.random.shuffle(idx_shuffle)
    images_shuffle = images[idx_shuffle]
    labels_shuffle = labels_one_hot[idx_shuffle]
    return images_shuffle, labels_shuffle


def relu(features):
    output = np.copy(features)
    output[features < 0] = 0
    return output


def drelu(features):
    output = np.copy(features)
    output[features > 0] = 1
    output[features <= 0] = 0
    return output


def softmax(data):
    exp_data = np.exp(data)
    sum_exp_data = np.sum(exp_data, axis=1, keepdims=True)
    return exp_data / sum_exp_data


def evaluate(images_test, labels, w1, b1, w2, b2):
    z1 = np.matmul(images_test, w1) + b1
    a1 = relu(z1)
    z2 = np.matmul(a1, w2) + b2
    softmax_z2 = softmax(z2)
    # argmax() return the index of max value
    labels_predict = softmax_z2.argmax(axis=1)
    labels_tmp = make_one_hot_labels(labels)
    is_right = labels_predict == labels
    loss = -np.sum(labels_tmp * np.log(softmax_z2))/len(labels)
    return loss, np.mean(is_right)


images_train = parse_mnist_images('train-images.idx3-ubyte')
images_test = parse_mnist_images('t10k-images.idx3-ubyte')
images_train = np.reshape(images_train, [images_train.shape[0], -1])
images_test = np.reshape(images_test, [images_test.shape[0], -1])
images_train = np.float32(images_train) / 255.0
images_test = np.float32(images_test) / 255.0
labels_train = parse_mnist_labels('train-labels.idx1-ubyte')
labels_test = parse_mnist_labels('t10k-labels.idx1-ubyte')
labels_train_one_hot = make_one_hot_labels(labels_train)

BATCH_SIZE = 100
EPOCH = 10
LEARNING_RATE = 0.01
num_train = images_train.shape[0]
single_image_size = images_train.shape[1]
w1 = uniform_random([single_image_size, 256], -0.002, 0.002)
b1 = np.zeros([1, 256])
w2 = uniform_random([256, 10], -0.002, 0.002)
b2 = np.zeros([1, 10])

train_loss = []
test_loss= []
test_acc = []

for ep in range(EPOCH):
    print('epoch', ep + 1)
    images_shuffle, labels_shuffle = shuffle_data(images_train,labels_train_one_hot)
    loss=0
    for i in range(0, num_train, BATCH_SIZE):
        # get a batch of data
        images_batch = images_shuffle[i:i + BATCH_SIZE, :]
        labels_batch = labels_shuffle[i:i + BATCH_SIZE, :]
        z1 = np.matmul(images_batch, w1) + b1
        a1 = relu(z1)
        z2 = np.matmul(a1, w2) + b2
        softmax_z2 = softmax(z2)
        # cross entropy loss, the loss is actually not used
        loss += -np.sum(labels_batch * np.log(softmax_z2))

        # backward propagation
        dz2 = softmax_z2 - labels_batch


        dw2 = np.matmul(a1.T, dz2)
        db2 = np.mean(dz2, axis=0)
        da1 = np.matmul(dz2, w2.T)
        dz1 = da1 * drelu(z1)

        dw1 = np.matmul(images_batch.T, dz1)
        db1 = np.mean(dz1, axis=0)

        # update paramters
        w1 -= LEARNING_RATE * dw1
        b1 -= LEARNING_RATE * db1
        w2 -= LEARNING_RATE * dw2
        b2 -= LEARNING_RATE * db2
   
    loss_, acc = evaluate(images_test, labels_test, w1, b1, w2, b2)
    train_loss.append(loss/num_train)
    test_loss.append(loss_)
    test_acc.append(acc)
    print('accuracy: ', acc)
print(train_loss)
print(test_loss)
print(test_acc)
# plt.plot(list(range(EPOCH)), train_loss)
# plt.plot(list(range(EPOCH)), test_acc)

plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.plot(list(range(EPOCH)), train_loss)
plt.savefig('train_loss.jpg',dpi=200)
plt.show()

plt.xlabel('epoch')
plt.ylabel('test_loss')
plt.plot(list(range(EPOCH)), test_loss)
plt.savefig('test_loss.jpg',dpi=200)
plt.show()

plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.plot(list(range(EPOCH)), test_acc)
plt.savefig('test_acc.jpg',dpi=200)
plt.show()