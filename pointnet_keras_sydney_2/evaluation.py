import os
import glob
import numpy as np
import tensorflow as tf
import csv
import h5py
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Concatenate, Multiply
from tensorflow.keras.layers import Flatten, Reshape, Dropout, Lambda
from tensorflow.keras.layers import Dense, Convolution1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.applications import ResNet101
import matplotlib.pyplot as plt

from model.model import PointTransNet, FeatureTransNet, Fusion
from model.model import mat_mul, Conv64, Conv2048, MLP
from evalu.function import IoU, mAP, f1_score, smoothL1


#DATA_DIR = '/data/projects/punim1332/sagarc/dataset/sydney-urban-objects-dataset/multi_class/classes'
DATA_DIR = '/Volumes/Sagar Seagate/Downloads/sydney-urban-objects-dataset/multi_class/classes'
NUM_POINTS = 1024
NUM_CLASSES = 9
BATCH_SIZE = 4
patience = 20

#tf.disable_v2_behavior()

def parse_dataset(num_points=1024):


    test_points = []
    test_labels = []
    class_map = {}
    folders = ['building', 'car', 'pedestrian', 'pillar', 'pole', 'traffic_sign', 'traffic_lights', 'tree', 'trunk']
    folders = [os.path.join(DATA_DIR,e) for e in folders]
    #folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in test_files:
            test_points.append(csv_vertex_parser(f, num_points))
            test_labels.append(i)

    return (
        np.array(test_points),
        np.array(test_labels),
        class_map,
    )

def csv_vertex_parser(path_to_csv_file, npoints):
    # Read the csv file
    with open(path_to_csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        num_vertices = len(rows)

    vertex_list = [np.expand_dims(np.array([float(rows[i][3]), float(rows[i][4]), float(rows[i][5])]), axis=0)
                   for i in range(num_vertices)]
    sample_points = np.vstack(vertex_list)

    ###padding
    if np.size(sample_points, 0) < npoints:
        delta_sample = np.ceil(npoints / np.size(sample_points, 0))
        temp_sample_points = []
        for _ in range(int(delta_sample)):
            temp_sample_points.append(sample_points)
        sample_points = np.vstack(temp_sample_points)

    ####pick up self.npoints points
    choice = np.random.choice(len(sample_points), npoints, replace=True)
    sample_points = sample_points[choice, :]

    return sample_points

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def evaluate():

    test_points, test_labels, CLASS_MAP = parse_dataset(
        NUM_POINTS
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    num_points = 1024
    input_points = Input(shape=(num_points, 3))
    # point transformation
    inp = PointTransNet(num_points = num_points)(input_points)
    input_T = Reshape((3,3))(inp)
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    # foward
    g = Conv64(num_points = num_points)(g)
    # feature transformation
    f = FeatureTransNet(num_points = num_points)(g)
    feature_T = Reshape((64, 64))(f)
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    # foward
    global_feature = Conv2048(num_points = num_points)(g)
    k = MLP()(global_feature)       #Classify the classes
    # output
    classes = Dense(NUM_CLASSES)(k)

    # model object
    model = Model(inputs=[input_points],
                  outputs=[classes])
    print(model.summary())

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)

    model.load_weights('/Volumes/Sagar Seagate/pointnet_keras_sydney_2/savedModels/PointNet22_SydneyMultiClass.h5')

    result = []

    loss = model.evaluate(test_dataset, verbose=0)
    print('Test Loss:', loss)

    # for i,data in enumerate(test_dataset, 0):
    #     points, target = data
    #     preds = model.predict(points)
    #     preds = tf.math.argmax(preds, -1)
    #     result.append(list(tf.make_ndarray(tf.make_tensor_proto(tf.math.equal(preds, target)))))

    # correct = 0
    # count = 0
    # for i in range(len(result)):
    #     for j in range(4):
    #         if result[i][j] == True:
    #             correct += 1
    #         count += 1
    #
    # accuracy = correct / float(count)
    #
    # print('Accuracy: ', accuracy)


if __name__ == '__main__':
    evaluate()
