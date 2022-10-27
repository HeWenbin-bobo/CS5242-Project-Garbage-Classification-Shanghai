from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.cluster import KMeans
from imutils import build_montages
import matplotlib.image as imgplt
from google.colab.patches import cv2_imshow
import os
import pandas as pd
import numpy as np
import cv2
import shutil
import sys
from tqdm import tqdm
import random

trash_path = "./Trash/"
image_crawler_save_path = './Image_crawler'
flickr_scrape_save_path = os.path.join(image_crawler_save_path, 'Flickr_scrape')

try:
    shutil.rmtree(trash_path)
    print(f"Removed past data in {trash_path}")
except OSError as e:
    print("Error: %s : %s" % (trash_path, e.strerror))

if not os.path.exists(trash_path):
    os.makedirs(trash_path)
    print(f"We create new directory {trash_path}")

class KMeansClass():

    def __init__(self, root_dir, label_dir, img_path, all_images, n_clusters=2):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.n_clusters = n_clusters
        self.img_path = img_path
        self.all_images = all_images
        self.clt = KMeans(n_clusters)
        # print(len(all_images))
        self.clt.fit(self.all_images)
        self.cluster_centers_ = self.clt.cluster_centers_
        self.imageLabelIDs = np.unique(self.clt.labels_)
        self.quantity = pd.Series(self.clt.labels_).value_counts()
        # print(self.quantity)

    def __len__(self):
        return len(self.all_images)

    def showKMeansResult(self):
        for labelID in self.imageLabelIDs:
            idxs = np.where(self.clt.labels_ == labelID)[0]
            idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
            show_box = []
            for i in idxs:
                img_name = self.img_path[i]
                image = cv2.imread(os.path.join(self.root_dir, self.label_dir, img_name))
                image = cv2.resize(image, (96, 96))
                show_box.append(image)
            montage = build_montages(show_box, (96, 96), (5, 5))[0]

            title = "Type {}".format(labelID)
            # cv2.imshow(title, montage)
            # print(title)
            cv2_imshow(montage)
            cv2.waitKey(0)

    def removeDifferentImages(self, trash_path):
        # remove the images in the cluster with lowest count
        for labelID in self.quantity.index.values[-1:]:
            # print(labelID)
            idxs = np.where(self.clt.labels_ == labelID)[0]
            # print(idxs)
            for i in idxs:
                img_name = self.img_path[i]
                remove_path = os.path.join(self.root_dir, self.label_dir, img_name)
                target_path = os.path.join(trash_path, self.label_dir)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    # print(f"We create new directory {target_path}")
                # print("We will move {} to {} !".format(remove_path, target_path))
                shutil.move(remove_path, target_path)


class DataCleaningLoadClassForKMeans(Dataset):
    
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)

    def KMeansFit(self, n_clusters=5):
        all_images = []
        for img_name in self.img_path:
            image = imgplt.imread(os.path.join(self.root_dir, self.label_dir, img_name))
            image = image / 255.0
            image = cv2.resize(image, (100, 100))
            image = image.reshape(-1,)
            all_images.append(image)
        
        self.kmeansClass = KMeansClass(self.root_dir, self.label_dir, self.img_path, all_images, n_clusters=5)

    def showKMeansResult(self):
        self.kmeansClass.showKMeansResult()

    def removeDifferentImages(self, trash_path):
        self.kmeansClass.removeDifferentImages(trash_path)

class DataCleaningResNet50(nn.Module):
    def __init__(self):
        super(DataCleaningResNet50, self).__init__()
        resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def KMeansFit(self, net, root_dir, label_dir, n_clusters=5):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        all_images = []
        for img_name in self.img_path:
            image = Image.open(os.path.join(self.root_dir, self.label_dir, img_name)).convert('RGB')
            image = transforms.Resize([224, 224])(image)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0)
            with torch.no_grad():
                image = net(image)
            image = image.reshape(-1, )
            all_images.append(image.detach().numpy())
        
        self.kmeansClass = KMeansClass(self.root_dir, self.label_dir, self.img_path, all_images, n_clusters=5)

    def showKMeansResult(self):
        self.kmeansClass.showKMeansResult()

    def removeDifferentImages(self, trash_path):
        self.kmeansClass.removeDifferentImages(trash_path)


if len(sys.argv) == 2 and sys.argv[1] == 'Kmeans':
    print('Start data cleaning through Kmeans...')
    for root, dirs, files in os.walk(flickr_scrape_save_path):
        for sub_dir in dirs:
    
            dataset_KMeans = DataCleaningLoadClassForKMeans(flickr_scrape_save_path, sub_dir)
    
            # img, label = dataset_KMeans[1]
            # display(img)
    
            dataset_KMeans.KMeansFit(n_clusters=8)
            # dataset_KMeans.showKMeansResult()
            dataset_KMeans.removeDifferentImages(trash_path)
    
            # break
else:
    print('Start data cleaning through ResNet50 and Kmeans...')
    dataCleaningResNet50 = DataCleaningResNet50()
    net = dataCleaningResNet50.eval()
    
    # check 10 times
    for i in tqdm(range(10)):
        # print(f"No.{i} check!")
    
        for root, dirs, files in os.walk(flickr_scrape_save_path):
            for sub_dir in dirs:
    
                dataCleaningResNet50.KMeansFit(net, flickr_scrape_save_path, sub_dir, n_clusters=8)
                # dataCleaningResNet50.showKMeansResult()
                dataCleaningResNet50.removeDifferentImages(trash_path)
    
                # break
        # print()


from skimage.metrics import structural_similarity as ssim

def removeImage(remove_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"We create new directory {target_path}")
    # print("We will move {} to {} !".format(remove_path, target_path))
    shutil.move(remove_path, target_path)


def list_all_files(root):
    files = []
    list = os.listdir(root)
    # os.listdir()方法：返回指定文件夹包含的文件或子文件夹名字的列表。该列表顺序以字母排序
    for i in range(len(list)):
        element = os.path.join(root, list[i])
        # 需要先使用python路径拼接os.path.join()函数，将os.listdir()返回的名称拼接成文件或目录的绝对路径再传入os.path.isdir()和os.path.isfile().
        if os.path.isdir(element):  # os.path.isdir()用于判断某一对象(需提供绝对路径)是否为目录
            # temp_dir = os.path.split(element)[-1]
            # os.path.split分割文件名与路径,分割为data_dir和此路径下的文件名，[-1]表示只取data_dir下的文件名
            files.append(list_all_files(element))

        elif os.path.isfile(element):
            files.append(element)
    # print('2',files)
    return files


def ssim_compare(img_files, ssim_value_threshold=0.8):
    count = 0
    for currIndex, filename in enumerate(img_files):
        if not os.path.exists(img_files[currIndex]):
            # print('not exist', img_files[currIndex])
            break
        img = cv2.imread(img_files[currIndex])
        # img = img / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.imread(img_files[currIndex + 1])
        # img1 = img1 / 255.0
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_LINEAR)
        #进行结构性相似度判断
        # ssim_value = _structural_similarity.structural_similarity(img,img1,multichannel=True)
        ssim_value = ssim(img, img1, multichannel=True)
        if ssim_value > ssim_value_threshold:
            #基数
            count += 1
            imgs_remove.append(img_files[currIndex + 1])
            # print('big_ssim:',img_files[currIndex], img_files[currIndex + 1], ssim_value)
        # 避免数组越界
        if currIndex+1 >= len(img_files)-1:
            break
    return count


# check 10 times
print("Start removing images with high similarity...")
for i in tqdm(range(10)):
    # print(f"No.{i} check!")
    for root, dirs, files in os.walk(flickr_scrape_save_path):
        for sub_dir in dirs:

            imgs_remove = []

            all_files = list_all_files(os.path.join(flickr_scrape_save_path, sub_dir)) #返回包含完整路径的所有图片名的列表
            # print('1',all_files)

            # for files in all_files:
            # 根据文件名排序，x.rfind('/')是从右边寻找第一个‘/’出现的位置，也就是最后出现的位置
            # 注意sort和sorted的区别，sort作用于原列表，sorted生成新的列表，且sorted可以作用于所有可迭代对象
            # all_files.sort(key = lambda x: x[x.rfind('/')+1:-4])#路径中包含“/”
            all_files.sort(key = lambda x: random.random())#random order
            # print(files)
            img_files = []
            for img in all_files:
                if img.endswith('.jpg'):
                    # 将所有图片名都放入列表中
                    img_files.append(img)
            count = ssim_compare(img_files)
            # print(img[:img.rfind('/')],"路径下删除的图片数量为：",count)
            for image in imgs_remove:
                removeImage(image, os.path.join(trash_path, sub_dir))
    # print()
