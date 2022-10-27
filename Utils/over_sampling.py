from gsmote import GeometricSMOTE
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset_save_path = "./Dataset/"
train_dataset_save_path = os.path.join(dataset_save_path, "train")

def images_df(dataset_save_path):
    all_images = []
    all_images_labels = []
    flag = True
    for root, dirs, files in os.walk(dataset_save_path):
        for label_dir in dirs:
            imgs = os.listdir(os.path.join(root, label_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            for img_name in imgs:
                image = Image.open(os.path.join(root, label_dir, img_name))
                image = np.array(image.convert("RGBA"))
                image = image / 255.0
                image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
                if flag:
                    # print(image.shape)
                    # plt.imshow(image)
                    flag = False
                image = image.reshape(-1,)
                all_images.append(image)
                all_images_labels.append(os.path.join(root, label_dir))
    
    X = pd.DataFrame(all_images)
    X = X.add_prefix('X')
    y = pd.DataFrame(all_images_labels, columns=['label'])
    y['label'] = y['label'].astype('category')
    labelencoder = LabelEncoder()
    y['label'] = labelencoder.fit_transform(y['label'])

    return X, y, labelencoder

print("Start loading all iamges...")
X_oversampling, y_oversampling, labelencoder = images_df(train_dataset_save_path)
# y_oversampling.groupby('label')['label'].count()

def plot_samples(X, y, title=None, n_subplots=None, channels=4):
    if not n_subplots:
        n_subplots = [1, np.unique(y).shape[0]]
    imshape = int(np.sqrt(X.shape[-1]/channels))
    fig, axes = plt.subplots(nrows=n_subplots[0], ncols=n_subplots[1], figsize=(20, 3))
    if title:
        fig.suptitle(title, fontsize=16)
    for i, val in enumerate(np.unique(y)):
        images = X[y == val]
        img = images.iloc[np.random.randint(images.shape[0])]
        img = (img*255).astype(np.uint8)
        if len(np.unique(y)) > 1:
            axes[i].imshow(np.reshape(img.values, (imshape, imshape, channels)))
            axes[i].set_title(str(val))
            axes[i].axis("off")
        else:
            axes.imshow(np.reshape(img.values, (imshape, imshape, channels)))
            axes.set_title(str(val))
            axes.axis("off")


def get_disjoin(X1, y1, X2, y2):
    """returns rows that do not belong to one of the two datasets"""
    if X1.shape[-1] != X2.shape[-1]:
        raise ValueError("Both arrays must have equal shape on axis 1.")

    if X1.shape[0] > X2.shape[0]:
        X_largest, y_largest, X_smallest, y_smallest = X1, y1, X2, y2
    else:
        X_largest, y_largest, X_smallest, y_smallest = X2, y2, X1, y1

    intersecting_vals = np.in1d(X_largest, X_smallest).reshape(X_largest.shape)
    disjoin_indexes = np.where(~np.all(intersecting_vals, axis=1))[0]
    return X_largest.iloc[disjoin_indexes], y_largest.iloc[disjoin_indexes]


def over_sampling_gsmote(X_train, y_train, strategies=["combined", "majority", "minority"], ds=[0, 0.5, 1], ts=[-1, 0, 1], k_neighbors=1):
    for strategy in strategies:
        X_gsmote_final = np.empty(shape=(0, X_train.shape[-1]))
        y_gsmote_final = np.empty(shape=(0))
        for d in ds:
            for t in ts:
                gsmote_sampling = GeometricSMOTE(
                    k_neighbors=k_neighbors,
                    deformation_factor=d,
                    truncation_factor=t,
                    n_jobs=-1,
                    selection_strategy=strategy,
                ).fit_resample(X_train, y_train)
                X_gsmote, y_gsmote = get_disjoin(
                    X_train, y_train, gsmote_sampling[0], gsmote_sampling[1]
                )
                X_gsmote_final = np.append(X_gsmote_final, X_gsmote, axis=0)
                y_gsmote_final = np.append(
                    y_gsmote_final, np.array([f"t={t}, d={d}"] * X_gsmote.shape[0]), axis=0
                )
        plot_samples(
            pd.DataFrame(X_gsmote_final),
            pd.Series(y_gsmote_final),
            f"Generated Using G-SMOTE: {strategy}",
        )
    return X_gsmote, y_gsmote

print("Start performing over_sampling...")
over_sampling_gsmote(X_oversampling, y_oversampling)
X_gsmote, y_gsmote = over_sampling_gsmote(X_oversampling, y_oversampling, strategies=["minority"], ds=[1], ts=[1], k_neighbors=4)

print("Start adding new image into {}".format(train_dataset_save_path))
dirs = labelencoder.inverse_transform(y_gsmote['label']).astype(str)
count = 0
last_subdir = ''
for index, subdir in tqdm(enumerate(dirs)):
    if last_subdir != subdir:
        count = 0
        # print()
    count += 1
    img = X_gsmote.iloc[index].values
    img = (img*255).astype(np.uint8)
    imshape = int(np.sqrt(img.shape[-1]/4))
    img = np.reshape(img, (imshape, imshape, 4))
    label = subdir.split('/')[-1]
    filename = label + "_" + str(count) + "(OverSampling)" + ".jpg"
    cv2.imwrite(os.path.join(subdir, filename), img)
    # print("We add {} into {} through oversampling technology".format(filename, subdir))
    last_subdir = subdir