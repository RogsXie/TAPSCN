import matplotlib
import numpy as np

matplotlib.use("Agg")
import datetime
import math
import os
from operator import truediv

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.utils.data as Data
from skimage import exposure
from sklearn.decomposition import PCA


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
    plt.close()


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(
        ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormaph(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([76, 188, 56]) / 255.0
        if item == 1:
            y[index] = np.array([128, 204, 42]) / 255.0
        if item == 2:
            y[index] = np.array([64, 138, 88]) / 255.0
        if item == 3:
            y[index] = np.array([56, 138, 62]) / 255.0
        if item == 4:
            y[index] = np.array([144, 72, 47]) / 255.0
        if item == 5:
            y[index] = np.array([114, 208, 210]) / 255.0
        if item == 6:
            y[index] = np.array([255, 255, 255]) / 255.0
        if item == 7:
            y[index] = np.array([201, 169, 206]) / 255.0
        if item == 8:
            y[index] = np.array([232, 33, 39]) / 255.0
        if item == 9:
            y[index] = np.array([121, 31, 36]) / 255.0
        if item == 10:
            y[index] = np.array([62, 99, 183]) / 255.0
        if item == 11:
            y[index] = np.array([223, 230, 49]) / 255.0
        if item == 12:
            y[index] = np.array([226, 134, 32]) / 255.0
        if item == 13:
            y[index] = np.array([80, 41, 137]) / 255.0
        if item == 14:
            y[index] = np.array([243, 99, 77]) / 255.0
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.0
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.0
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.0
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.0
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.0
    return y


def list_to_colormapm(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 128, 1]) / 255.0
        if item == 1:
            y[index] = np.array([0, 255, 1]) / 255.0
        if item == 2:
            y[index] = np.array([2, 1, 203]) / 255.0
        if item == 3:
            y[index] = np.array([254, 203, 0]) / 255.0
        if item == 4:
            y[index] = np.array([252, 0, 49]) / 255.0
        if item == 5:
            y[index] = np.array([114, 208, 210]) / 255.0
        if item == 6:
            y[index] = np.array([102, 0, 205]) / 255.0
        if item == 7:
            y[index] = np.array([254, 126, 151]) / 255.0
        if item == 8:
            y[index] = np.array([201, 102, 0]) / 255.0
        if item == 9:
            y[index] = np.array([254, 254, 0]) / 255.0
        if item == 10:
            y[index] = np.array([204, 66, 100]) / 255.0
        if item == 11:
            y[index] = np.array([223, 230, 49]) / 255.0
        if item == 12:
            y[index] = np.array([226, 134, 32]) / 255.0
        if item == 13:
            y[index] = np.array([80, 41, 137]) / 255.0
        if item == 14:
            y[index] = np.array([243, 99, 77]) / 255.0
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.0
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.0
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.0
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.0
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.0
    return y


def list_to_colormapt(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 128, 1]) / 255.0
        if item == 1:
            y[index] = np.array([78, 200, 237]) / 255.0
        if item == 2:
            y[index] = np.array([59, 86, 167]) / 255.0
        if item == 3:
            y[index] = np.array([254, 210, 13]) / 255.0
        if item == 4:
            y[index] = np.array([238, 52, 37]) / 255.0
        if item == 5:
            y[index] = np.array([125, 21, 22]) / 255.0
        if item == 6:
            y[index] = np.array([121, 21, 22]) / 255.0
        if item == 7:
            y[index] = np.array([201, 169, 206]) / 255.0
        if item == 8:
            y[index] = np.array([232, 33, 39]) / 255.0
        if item == 9:
            y[index] = np.array([121, 31, 36]) / 255.0
        if item == 10:
            y[index] = np.array([62, 99, 183]) / 255.0
        if item == 11:
            y[index] = np.array([223, 230, 49]) / 255.0
        if item == 12:
            y[index] = np.array([226, 134, 32]) / 255.0
        if item == 13:
            y[index] = np.array([80, 41, 137]) / 255.0
        if item == 14:
            y[index] = np.array([243, 99, 77]) / 255.0
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.0
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.0
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.0
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.0
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.0
    return y


def list_to_colormaph18(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([76, 188, 56]) / 255.0
        if item == 1:
            y[index] = np.array([180, 210, 80]) / 255.0
        if item == 2:
            y[index] = np.array([64, 138, 88]) / 255.0
        if item == 3:
            y[index] = np.array([34, 100, 34]) / 255.0
        if item == 4:
            y[index] = np.array([94, 160, 50]) / 255.0
        if item == 5:
            y[index] = np.array([160, 110, 60]) / 255.0
        if item == 6:
            y[index] = np.array([60, 160, 230]) / 255.0
        if item == 7:
            y[index] = np.array([240, 180, 180]) / 255.0
        if item == 8:
            y[index] = np.array([210, 60, 60]) / 255.0
        if item == 9:
            y[index] = np.array([160, 160, 160]) / 255.0
        if item == 10:
            y[index] = np.array([220, 220, 220]) / 255.0
        if item == 11:
            y[index] = np.array([255, 255, 200]) / 255.0
        if item == 12:
            y[index] = np.array([100, 100, 100]) / 255.0
        if item == 13:
            y[index] = np.array([60, 60, 60]) / 255.0
        if item == 14:
            y[index] = np.array([130, 60, 200]) / 255.0
        if item == 15:
            y[index] = np.array([240, 220, 60]) / 255.0
        if item == 16:
            y[index] = np.array([210, 150, 50]) / 255.0
        if item == 17:
            y[index] = np.array([80, 220, 220]) / 255.0
        if item == 18:
            y[index] = np.array([220, 80, 220]) / 255.0
        if item == 19:
            y[index] = np.array([240, 120, 40]) / 255.0
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.0
    return y


def list_to_colormapaug(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([124, 209, 86]) / 255.0
        if item == 1:
            y[index] = np.array([34, 113, 34]) / 255.0
        if item == 2:
            y[index] = np.array([100, 100, 100]) / 255.0
        if item == 3:
            y[index] = np.array([220, 80, 80]) / 255.0
        if item == 4:
            y[index] = np.array([60, 160, 230]) / 255.0
        if item == 5:
            y[index] = np.array([160, 80, 200]) / 255.0
        if item == 6:
            y[index] = np.array([195, 155, 90]) / 255.0
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.0
    return y


def create_colored_classified_image(gt_mat, dataset):
    data = gt_mat - 1
    if dataset == "Houston":
        colored_image = list_to_colormaph(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Trento":
        colored_image = list_to_colormapt(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Muufl":
        colored_image = list_to_colormapm(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Augsburg":
        colored_image = list_to_colormapaug(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Houston18":
        colored_image = list_to_colormaph18(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    output_image_path = os.path.join(dataset, "gt_colormap.png")
    plt.imshow(colored_image)
    plt.axis("off")
    plt.savefig(
        output_image_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
    )
    plt.close()


def all_create_colored_classified_image(gt_mat, dataset):
    data = gt_mat - 1
    if dataset == "Houston":
        colored_image = list_to_colormaph(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Trento":
        colored_image = list_to_colormapt(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Muufl":
        colored_image = list_to_colormapm(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Augsburg":
        colored_image = list_to_colormapaug(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    elif dataset == "Houston18":
        colored_image = list_to_colormaph18(data.flatten()).reshape(
            data.shape[0], data.shape[1], 3
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not os.path.exists(dataset):
        os.makedirs(dataset)
    output_image_path = os.path.join(dataset, "all_colormap.png")
    plt.imshow(colored_image)
    plt.axis("off")
    plt.savefig(
        output_image_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
    )
    plt.close()


def _get_times_new_roman_prop(size=20):
    candidates = fm.findSystemFonts(fontpaths=None, fontext="ttf")
    tnr_path = None
    for path in candidates:
        basename = os.path.basename(path).lower()
        if (
            "timesnewroman" in basename.replace(" ", "")
            or "times new roman" in basename
            or basename
            in (
                "times.ttf",
                "timesbd.ttf",
                "timesi.ttf",
                "timesbi.ttf",
                "times new roman.ttf",
            )
        ):
            tnr_path = path
            break
    if tnr_path:
        return fm.FontProperties(fname=tnr_path, size=size)
    else:
        return fm.FontProperties(family="serif", size=size)


def plot_category_colors(dataset):
    dataset_folder = f"./{dataset}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    font_prop = _get_times_new_roman_prop(size=20)
    if dataset == "Houston":
        categories = [
            "Healthy Grass",
            "Stressed Grass",
            "Synthetic Grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]
        category_ids = np.arange(0, len(categories))
        color_map = list_to_colormaph(category_ids)
    elif dataset == "Trento":
        categories = [
            "Apple Trees",
            "Buildings",
            "Ground",
            "Woods",
            "Vineyard",
            "Roads",
        ]
        category_ids = np.arange(0, len(categories))
        color_map = list_to_colormapt(category_ids)
    elif dataset == "Muufl":
        categories = [
            "Trees",
            "Mostly Grass",
            "Mixed Ground Surface",
            "Dirt and Sand",
            "Roads",
            "Water",
            "Buildings Shadow",
            "Buildings",
            "Sidewalk",
            "Yellow Curb",
            "Cloth Panels",
        ]
        category_ids = np.arange(0, len(categories))
        color_map = list_to_colormapm(category_ids)
    elif dataset == "Augsburg":
        categories = [
            "Low Plants",
            "Trees",
            "Asphalt",
            "Rooftops",
            "Water",
            "Railway",
            "Bare Soil",
        ]
        category_ids = np.arange(0, len(categories))
        color_map = list_to_colormapaug(category_ids)
    elif dataset == "Houston18":
        categories = [
            "Healthy Grass",
            "Stressed Grass",
            "Artificial Turf",
            "Evergreen Trees",
            "Deciduous Trees",
            "Bare Earth",
            "Water",
            "Residential Buildings",
            "Non-residential Buildings",
            "Roads",
            "Sidewalks",
            "Crosswalks",
            "Major Thoroughfares",
            "Highways",
            "Railways",
            "Paved Parking Lots",
            "Unpaved Parking Lots",
            "Cars",
            "Trains",
            "Stadium Seats",
        ]
        category_ids = np.arange(0, len(categories))
        color_map = list_to_colormaph18(category_ids)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    height = len(categories)
    width = 200
    color_bar_height = 50
    spacing = 10
    image = np.ones(((color_bar_height + spacing) * height, width, 3))
    for i in range(height):
        start_row = i * (color_bar_height + spacing)
        image[start_row : start_row + color_bar_height, :, :] = color_map[i]
        image[start_row : start_row + color_bar_height, 0, :] = 0
        image[start_row : start_row + color_bar_height, -1, :] = 0
        image[start_row, :, :] = 0
        image[start_row + color_bar_height - 1, :, :] = 0
    save_path = os.path.join(dataset_folder, "category_colors.png")
    plt.figure(figsize=(5, max(5, height)))
    plt.imshow(image)
    plt.axis("off")
    for i in range(height):
        y_pos = i * (color_bar_height + spacing) + color_bar_height / 2
        x_pos = width / 2
        bg = color_map[i]
        luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        txt_color = "white" if luminance < 0.45 else "black"
        plt.text(
            x_pos,
            y_pos,
            categories[i],
            va="center",
            ha="center",
            fontproperties=font_prop,
            color=txt_color,
        )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True)
    plt.close()
    print(f"Image saved at {save_path}")


def process_and_display_image(hsi_data, lidar_data, mode, dataset):
    dataset_folder = f"./{dataset}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if dataset == "Houston":
        if mode == "single":
            data = lidar_data.astype(np.float32)
            data_normalized = 255 * (data - data.min()) / (data.max() - data.min())
            data_normalized = data_normalized.astype(np.uint8)
            save_path = os.path.join(dataset_folder, "classified_image_single.png")
            plt.imshow(data_normalized, cmap="gray")
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "dual":
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            rgb_image = np.dstack((data[:, :, 30], data[:, :, 20], data[:, :, 10]))
            for i in range(3):
                rgb_image[:, :, i] = exposure.equalize_hist(rgb_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "both":
            data_lidar = lidar_data
            data_lidar_normalized = (
                255
                * (data_lidar - data_lidar.min())
                / (data_lidar.max() - data_lidar.min())
            )
            data_lidar_normalized = data_lidar_normalized.astype(np.uint8)
            lidar_save_path = os.path.join(
                dataset_folder, "classified_image_single.png"
            )
            plt.imshow(data_lidar_normalized, cmap="gray")
            plt.axis("off")
            plt.savefig(
                lidar_save_path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
                transparent=True,
            )
            plt.close()
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            reshaped_data = data.reshape((-1, data.shape[2]))
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(reshaped_data)
            pca_image = pca_result.reshape((data.shape[0], data.shape[1], 3))
            for i in range(3):
                pca_image[:, :, i] = exposure.equalize_hist(pca_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(pca_image)
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
    elif dataset == "Trento":
        if mode == "single":
            data = lidar_data.astype(np.float32)
            data_normalized = 255 * (data - data.min()) / (data.max() - data.min())
            data_normalized = data_normalized.astype(np.uint8)
            save_path = os.path.join(dataset_folder, "classified_image_single.png")
            plt.imshow(data_normalized, cmap="gray")
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "dual":
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            rgb_image = np.dstack((data[:, :, 30], data[:, :, 20], data[:, :, 10]))
            for i in range(3):
                rgb_image[:, :, i] = exposure.equalize_hist(rgb_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "both":
            data_lidar = lidar_data
            data_lidar_normalized = (
                255
                * (data_lidar - data_lidar.min())
                / (data_lidar.max() - data_lidar.min())
            )
            data_lidar_normalized = data_lidar_normalized.astype(np.uint8)
            lidar_save_path = os.path.join(
                dataset_folder, "classified_image_single.png"
            )
            plt.imshow(data_lidar_normalized, cmap="gray")
            plt.axis("off")
            plt.savefig(
                lidar_save_path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
                transparent=True,
            )
            plt.close()
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            reshaped_data = data.reshape((-1, data.shape[2]))
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(reshaped_data)
            pca_image = pca_result.reshape((data.shape[0], data.shape[1], 3))
            for i in range(3):
                pca_image[:, :, i] = exposure.equalize_hist(pca_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(pca_image)
            plt.axis("off")
            plt.savefig(
                lidar_save_path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
                transparent=True,
            )
            plt.close()
    elif dataset == "Muufl":
        if mode == "single":
            data = lidar_data.astype(np.float32)
            data_normalized_1 = (
                255
                * (data[:, :, 0] - data[:, :, 0].min())
                / (data[:, :, 0].max() - data[:, :, 0].min())
            )
            data_normalized_2 = (
                255
                * (data[:, :, 1] - data[:, :, 1].min())
                / (data[:, :, 1].max() - data[:, :, 1].min())
            )
            data_normalized_1 = data_normalized_1.astype(np.uint8)
            data_normalized_2 = data_normalized_2.astype(np.uint8)
            gray_image = (0.5 * data_normalized_1 + 0.5 * data_normalized_2).astype(
                np.uint8
            )
            save_path = os.path.join(dataset_folder, "classified_image_gray.png")
            plt.imshow(gray_image, cmap="gray")
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "dual":
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            rgb_image = np.dstack((data[:, :, 30], data[:, :, 20], data[:, :, 10]))
            for i in range(3):
                rgb_image[:, :, i] = exposure.equalize_hist(rgb_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
        elif mode == "both":
            data = lidar_data.astype(np.float32)
            data_normalized_1 = (
                255
                * (data[:, :, 0] - data[:, :, 0].min())
                / (data[:, :, 0].max() - data[:, :, 0].min())
            )
            data_normalized_2 = (
                255
                * (data[:, :, 1] - data[:, :, 1].min())
                / (data[:, :, 1].max() - data[:, :, 1].min())
            )
            data_normalized_1 = data_normalized_1.astype(np.uint8)
            data_normalized_2 = data_normalized_2.astype(np.uint8)
            gray_image = (0.5 * data_normalized_1 + 0.5 * data_normalized_2).astype(
                np.uint8
            )
            lidar_save_path = os.path.join(
                dataset_folder, "classified_image_single.png"
            )
            plt.imshow(gray_image, cmap="gray")
            plt.axis("off")
            plt.savefig(
                lidar_save_path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
                transparent=True,
            )
            plt.close()
            data = hsi_data
            for i in range(data.shape[2]):
                mean = np.mean(data[:, :, i])
                std = np.std(data[:, :, i])
                data[:, :, i] = (data[:, :, i] - mean) / std
            reshaped_data = data.reshape((-1, data.shape[2]))
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(reshaped_data)
            pca_image = pca_result.reshape((data.shape[0], data.shape[1], 3))
            for i in range(3):
                pca_image[:, :, i] = exposure.equalize_hist(pca_image[:, :, i])
            save_path = os.path.join(dataset_folder, "classified_image_dual.png")
            plt.imshow(pca_image)
            plt.axis("off")
            plt.savefig(
                save_path, bbox_inches="tight", pad_inches=0, dpi=300, transparent=True
            )
            plt.close()
