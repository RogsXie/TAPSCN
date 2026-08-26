import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from scipy import io
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def nor_pca(Data, Data2, ispca=True):
    [m, n, l] = Data.shape
    for i in range(l):
        minimal = Data[:, :, i].min()
        maximal = Data[:, :, i].max()
        Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)
    minimal = Data2.min()
    maximal = Data2.max()
    Data2 = (Data2 - minimal) / (maximal - minimal)
    if ispca is True:
        NC = 20
        PC = np.reshape(Data, (m * n, l))
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)
        PC = np.reshape(PC, (m, n, NC))
    else:
        NC = l
        PC = Data
    return (PC, Data2, NC)


def load_data(data_set_name, data_path1="C:/VLM/four/dataset"):
    if data_set_name == "Muufl":
        data1 = io.loadmat(os.path.join(data_path1, "MUUFL", "HSI.mat"))["HSI"]
        data2 = io.loadmat(os.path.join(data_path1, "MUUFL", "LiDAR.mat"))["LiDAR"]
        labels = io.loadmat(os.path.join(data_path1, "MUUFL", "gt.mat"))["gt"]
        train = io.loadmat(os.path.join(data_path1, "MUUFL", "mask_train_150"))[
            "mask_train"
        ]
        test = io.loadmat(os.path.join(data_path1, "MUUFL", "mask_test_150"))[
            "mask_test"
        ]
    if data_set_name == "Trento":
        data1 = io.loadmat(os.path.join(data_path1, "Trento", "HSI.mat"))["HSI"]
        data2 = io.loadmat(os.path.join(data_path1, "Trento", "LiDAR.mat"))["LiDAR"]
        labels = io.loadmat(os.path.join(data_path1, "Trento", "gt.mat"))["gt"]
        train = io.loadmat(os.path.join(data_path1, "Trento", "TRLabel"))["TRLabel"]
        test = io.loadmat(os.path.join(data_path1, "Trento", "TSLabel"))["TSLabel"]
    if data_set_name == "Houston":
        data1 = io.loadmat(
            os.path.join(data_path1, "Houston2013_Data", "Houston2013_HSI.mat")
        )["HSI"]
        data2 = io.loadmat(
            os.path.join(data_path1, "Houston2013_Data", "Houston2013_DSM.mat")
        )["DSM"]
        labels = io.loadmat(os.path.join(data_path1, "Houston2013_Data", "gt.mat"))[
            "gt"
        ]
        train = io.loadmat(
            os.path.join(data_path1, "Houston2013_Data", "Houston2013_TR")
        )["TR_map"]
        test = io.loadmat(
            os.path.join(data_path1, "Houston2013_Data", "Houston2013_TE")
        )["TE_map"]
    if data_set_name == "Houston18":
        data1 = io.loadmat(os.path.join(data_path1, "Houston2018", "houston_hsi.mat"))[
            "houston_hsi"
        ]
        data2 = io.loadmat(
            os.path.join(data_path1, "Houston2018", "houston_lidar.mat")
        )["houston_lidar"]
        labels = io.loadmat(os.path.join(data_path1, "Houston2018", "houston_gt.mat"))[
            "houston_gt"
        ]
    if data_set_name == "Augsburg":
        data1 = io.loadmat(os.path.join(data_path1, "Augsburg", "data_HS_LR.mat"))[
            "data_HS_LR"
        ]
        data2 = io.loadmat(os.path.join(data_path1, "Augsburg", "data_DSM.mat"))[
            "data_DSM"
        ]
        labels = io.loadmat(os.path.join(data_path1, "Augsburg", "gt.mat"))["gt"]
        train = io.loadmat(os.path.join(data_path1, "Augsburg", "TrainImage.mat"))[
            "TrainImage"
        ]
        test = io.loadmat(os.path.join(data_path1, "Augsburg", "TestImage.mat"))[
            "TestImage"
        ]
    return (data1, data2, labels, train, test)


def loadtrandte_data(data_set_name):
    if data_set_name == "Muufl":
        data1 = io.loadmat(os.path.join("MUUFL", "train_labels"))["train_labels"]
        data2 = io.loadmat(os.path.join("MUUFL", "test_labels"))["test_labels"]
    if data_set_name == "Trento":
        data1 = io.loadmat(os.path.join("Trento", "train_labels"))["train_labels"]
        data2 = io.loadmat(os.path.join("Trento", "test_labels"))["test_labels"]
    if data_set_name == "Houston":
        data1 = io.loadmat(os.path.join("Houston", "train_labels"))["train_labels"]
        data2 = io.loadmat(os.path.join("Houston", "test_labels"))["test_labels"]
    return (data1, data2)


def normalize(input2):
    input2_normalize = np.zeros(input2.shape, dtype=np.float32)
    if len(input2.shape) == 2:
        input2_max = np.max(input2)
        input2_min = np.min(input2)
        input2_normalize = (input2 - input2_min) / (input2_max - input2_min)
    elif len(input2.shape) == 3:
        for i in range(input2.shape[2]):
            input2_max = np.max(input2[:, :, i])
            input2_min = np.min(input2[:, :, i])
            input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (
                input2_max - input2_min
            )
    else:
        raise ValueError(
            "Input data must be 2D (height, width) or 3D (height, width, bands)."
        )
    return input2_normalize


def standardization(data):
    if len(data.shape) == 2:
        height, width = data.shape
        data = np.reshape(data, [height * width, 1])
        data = preprocessing.StandardScaler().fit_transform(data)
        data = np.reshape(data, [height, width])
    elif len(data.shape) == 3:
        height, width, bands = data.shape
        data = np.reshape(data, [height * width, bands])
        data = preprocessing.StandardScaler().fit_transform(data)
        data = np.reshape(data, [height, width, bands])
    else:
        raise ValueError(
            "Input data must be 2D (height, width) or 3D (height, width, bands)."
        )
    return data


def ImageStretching(image):
    channels = image.shape[2]
    band_list = []
    for i in range(channels):
        band_data = image[:, :, i]
        band_min = np.percentile(band_data, 2)
        band_max = np.percentile(band_data, 98)
        band_data = (band_data - band_min) / (band_max - band_min)
        band_list.append(band_data)
    image_data = np.stack(band_list, axis=-1)
    image_data = np.clip(image_data, 0, 1)
    image_data = (image_data * 255).astype(np.uint8)
    image_data = np.uint8(image_data)
    return image_data


def padpatch(Data1, Data2, Data3, patchsize, pad_width):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)
    x1 = Data1
    x2 = Data2
    x3 = Data3
    x1_pad = np.empty(
        (m1 + patchsize + offset, n1 + patchsize + offset, l1), dtype="float32"
    )
    x2_pad = np.empty(
        (m2 + patchsize + offset, n2 + patchsize + offset, l2), dtype="float32"
    )
    x3_pad = np.empty(
        (m3 + patchsize + offset, n3 + patchsize + offset, l3), dtype="float32"
    )
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x3_pad[:, :, i] = temp2
    return (x1_pad, x2_pad, x3_pad)


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    (
        all_label_index_dict,
        train_label_index_dict,
        val_label_index_dict,
        test_label_index_dict,
    ) = ({}, {}, {}, {})
    (
        all_label_index_list,
        train_label_index_list,
        val_label_index_list,
        test_label_index_list,
    ) = ([], [], [], [])
    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)
        np.random.shuffle(cls_index)
        if Flag == 0:
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)
            val_index_flag = max(int(ratio_list[1] * len(cls_index)), 1)
        elif Flag == 1:
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15
            val_index_flag = num_list[1]
        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:][val_index_flag:])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:][:val_index_flag])
        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]
    return (
        train_label_index_list,
        val_label_index_list,
        test_label_index_list,
        all_label_index_list,
    )


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def split_train_test_labels(
    label_matrix, num_samples_per_class, dataset, random_seed=None
):
    np.random.seed(random_seed)
    num_classes = label_matrix.max()
    train_indices = []
    test_indices = []
    for class_label in range(1, num_classes + 1):
        class_indices = np.where(label_matrix == class_label)
        class_indices = np.array(class_indices).T
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:num_samples_per_class])
        test_indices.extend(class_indices[num_samples_per_class:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    train_labels = np.zeros_like(label_matrix)
    test_labels = np.copy(label_matrix)
    for idx in train_indices:
        train_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    for idx in test_indices:
        test_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    test_labels[train_indices[:, 0], train_indices[:, 1]] = 0
    dataset_folder = f"./{dataset}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    train_file_path = os.path.join(dataset_folder, "train_labels.mat")
    test_file_path = os.path.join(dataset_folder, "test_labels.mat")
    sio.savemat(train_file_path, {"train_labels": train_labels})
    sio.savemat(test_file_path, {"test_labels": test_labels})
    return (train_labels, test_labels)


def trainone_patch(Data1, patchsize, pad_width, Label, ALL_Indices):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    x1 = Data1
    x1_pad = np.empty(
        (m1 + patchsize + offset, n1 + patchsize + offset, l1), dtype="float32"
    )
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x1_pad[:, :, i] = temp2
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype="float32")
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return (TrainPatch1, TrainLabel)


def traintwo_patch(Data1, Data2, patchsize, pad_width, Label, ALL_Indices):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    x1 = Data1
    x2 = Data2
    x1_pad = np.empty(
        (m1 + patchsize + offset, n1 + patchsize + offset, l1), dtype="float32"
    )
    x2_pad = np.empty(
        (m2 + patchsize + offset, n2 + patchsize + offset, l2), dtype="float32"
    )
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x2_pad[:, :, i] = temp2
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype="float32")
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype="float32")
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return (TrainPatch1, TrainPatch2, TrainLabel)


def trainthird_patch(Data1, Data2, Data3, patchsize, pad_width, Label, ALL_Indices):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)
    x1 = Data1
    x2 = Data2
    x3 = Data3
    x1_pad = np.empty(
        (m1 + patchsize + offset, n1 + patchsize + offset, l1), dtype="float32"
    )
    x2_pad = np.empty(
        (m2 + patchsize + offset, n2 + patchsize + offset, l2), dtype="float32"
    )
    x3_pad = np.empty(
        (m3 + patchsize + offset, n3 + patchsize + offset, l3), dtype="float32"
    )
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width + offset, "symmetric")
        x3_pad[:, :, i] = temp2
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype="float32")
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype="float32")
    TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype="float32")
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patch3 = x3_pad[
            ind3[i] - pad_width : ind3[i] + pad_width + offset,
            ind4[i] - pad_width : ind4[i] + pad_width + offset,
            :,
        ]
        patch3 = np.transpose(patch3, (2, 0, 1))
        TrainPatch3[i, :, :, :] = patch3
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return (TrainPatch1, TrainPatch2, TrainPatch3, TrainLabel)


def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SEEDS(object):
    def __init__(
        self,
        LiDAR,
        n_segments,
        num_levels=2,
        prior=1,
        histogram_bins=5,
        num_iterations=4,
    ):
        self.n_segments = n_segments
        self.num_levels = num_levels
        self.prior = prior
        self.histogram_bins = histogram_bins
        self.num_iterations = num_iterations
        self.data = LiDAR

    def SEEDS_superpixel(self, I):
        I_new = np.array(I[:, :, 0:3], np.float32).copy()
        height, width, channels = I_new.shape
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            width,
            height,
            channels,
            int(self.n_segments),
            num_levels=self.num_levels,
            prior=self.prior,
            histogram_bins=self.histogram_bins,
        )
        seeds.iterate(I_new, self.num_iterations)
        segments = seeds.getLabels()
        return segments

    def get_Q_and_S_and_Segments(self, img):
        h, w, d = img.shape
        segments = self.SEEDS_superpixel(img)
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
        out = mark_boundaries(img[:, :, 0], segments)
        plt.figure()
        plt.imshow(out)
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        self.S = S
        self.Q = Q
        return (Q, S, self.segments)

    def get_A(self, sigma: float):
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        h, w = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i : i + 2, j : j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma**2)
                    A[idx1, idx2] = A[idx2, idx1] = diss
        return A


class Superpixel_Seg(object):
    def __init__(self, scale):
        self.scale = scale

    def SGB(self, data):
        height, width = data.shape[:2]
        n_segments_init = height * width / self.scale
        print(f"Initial number of segments: {n_segments_init}")
        myseeds = SEEDS(
            data,
            n_segments_init,
            num_levels=2,
            prior=1,
            histogram_bins=5,
            num_iterations=4,
        )
        Q, S, Segments = myseeds.get_Q_and_S_and_Segments(data)
        A = myseeds.get_A(sigma=10)
        return (Q, S, A, Segments)
