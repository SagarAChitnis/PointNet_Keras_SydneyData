import torch
import torch.utils.data
import math
import random
import numpy as np
import os
import os.path as osp
import csv

class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path, npoint, test):
        self.test = test
        self.npoints = npoint
        # Build path list
        self.input_pairs, self.gt_key = self.create_input_list(
            dataset_root_path, test)
    def __len__(self):
        return len(self.input_pairs)

    def __getitem__(self, idx):
        # Select the path
        path, label = self.input_pairs[idx]
        # Parse the vertices from the file
        vertices = self.off_vertex_parser(path)
        if not self.test:
            vertices = self.augment_data(vertices)
        # Convert numpy format to torch variable
        return torch.from_numpy(vertices.astype(np.float32)), torch.from_numpy(np.array(label).astype(np.float32))

    def get_gt_key(self):
        return self.gt_key

    def create_input_list(self, dataset_root_path, test):
        input_pairs = []

        #  List of tuples grouping a label with a class
        gt_key = os.listdir(dataset_root_path)
        if '.DS_Store' in gt_key:
            gt_key.remove('.DS_Store')
        for idx, obj in enumerate(gt_key):
            if test:
                path_to_files = osp.join(dataset_root_path, obj, 'test')
            else:
                path_to_files = osp.join(dataset_root_path, obj, 'train')
            files = os.listdir(path_to_files)
            if '.DS_Store' in files:
                files.remove('.DS_Store')
            filepaths = [(osp.join(path_to_files, file), idx)
                         for file in files]
            input_pairs = input_pairs + filepaths

        return input_pairs, gt_key

    def augment_data(self, vertices):

        # Random rotation about the Z-axis
        rot = random.uniform(0, 2 * math.pi)
        rotation_matrix = [[math.cos(rot), math.sin(rot), 0],
                           [-math.sin(rot), math.cos(rot), 0],
                           [0, 0, 1]]
        vertices = np.dot(vertices, rotation_matrix)
        # Add Gaussian noise with standard deviation of 0.2
        vertices += np.random.normal(scale=0.02)
        return vertices

    def off_vertex_parser(self, path_to_off_file):
        # Read the OFF file
        with open(path_to_off_file, 'r') as f:
            contents = f.readlines()

        if contents[0].strip().lower() != 'off':
            num_vertices = int(contents[0].strip()[3:].split(' ')[0])
            start_line = 1
        else:
            num_vertices = int(contents[1].strip().split(' ')[0])
            start_line = 2

        vertex_list = [np.expand_dims(np.array(list(map(float, contents[i].strip().split(' ')))), axis=0) for i in
                       range(start_line, start_line + num_vertices)]
        sample_points = np.vstack(vertex_list)

        ###There are not enough vertex in some samples
        if np.size(sample_points, 0) < self.npoints:
            delta_sample = np.ceil(self.npoints / np.size(sample_points, 0))
            temp_sample_points = []
            for _ in range(int(delta_sample)):
                temp_sample_points.append(sample_points)
            sample_points = np.vstack(temp_sample_points)

        ####pick up self.npoints points
        choice = np.random.choice(len(sample_points), self.npoints, replace=True)
        sample_points = sample_points[choice, :]

        return sample_points

class SydneyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root_path, npoint, test=False):
        self.test = test
        self.npoints = npoint
        # Build path list
        self.input_pairs, self.gt_key = self.create_input_list(dataset_root_path)
    def __len__(self):
        return len(self.input_pairs)

    def __getitem__(self, idx):
        # Select the path
        path, label = self.input_pairs[idx]
        # Parse the vertices from the file
        vertices = self.csv_vertex_parser(path)
        if not self.test:
            vertices = self.augment_data_rotation(vertices)
        # Convert numpy format to torch variable
        return torch.from_numpy(vertices.astype(np.float32)), torch.from_numpy(np.array(label).astype(np.float32))

    def get_gt_key(self):
        return self.gt_key

    def create_input_list(self, dataset_root_path):
        input_pairs = []

        #  List of tuples grouping a label with a class
        gt_key = os.listdir(dataset_root_path)
        gt_key.remove('.DS_Store')
        for idx, obj in enumerate(gt_key):
            files = os.listdir(osp.join(dataset_root_path, obj))
            filepaths = [(osp.join(osp.join(dataset_root_path, obj), file), idx)
                         for file in files]
            input_pairs = input_pairs + filepaths

        return input_pairs, gt_key

    def augment_data_rotation(self, vertices):

        # Normalize
        vertices = vertices - np.mean(vertices, axis= 0) # Subtracting mean
        vertices /= np.max(np.linalg.norm(vertices, axis= 1))  # normalize into unit sphere

        # Random rotation about the Z-axis
        rot = random.uniform(0, 2 * math.pi)
        rotation_matrix = [[math.cos(rot), math.sin(rot), 0],
                           [-math.sin(rot), math.cos(rot), 0],
                           [0, 0, 1]]
        vertices = np.dot(vertices, rotation_matrix)

        # Add Gaussian noise with standard deviation of 0.2
        vertices += np.random.normal(scale=0.02)
        return vertices

    def csv_vertex_parser(self, path_to_csv_file):
        # Read the csv file
        with open(path_to_csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            num_vertices = len(rows)

        vertex_list = [np.expand_dims(np.array([float(rows[i][3]), float(rows[i][4]), float(rows[i][5])]), axis=0)
                       for i in range(num_vertices)]
        sample_points = np.vstack(vertex_list)

        ###padding
        if np.size(sample_points, 0) < self.npoints:
            delta_sample = np.ceil(self.npoints / np.size(sample_points, 0))
            temp_sample_points = []
            for _ in range(int(delta_sample)):
                temp_sample_points.append(sample_points)
            sample_points = np.vstack(temp_sample_points)

        ####pick up self.npoints points
        choice = np.random.choice(len(sample_points), self.npoints, replace=True)
        sample_points = sample_points[choice, :]

        return sample_points

