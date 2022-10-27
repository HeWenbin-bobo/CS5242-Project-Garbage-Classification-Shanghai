from flickrapi import FlickrAPI
import os
import pandas as pd
import sys
import random
import shutil
from tqdm import tqdm
import time
import requests
import numpy as np

print("Start using Flickr_crawler...")

key = 'ada677c758306224de32e21219a644f5'
secret = 'f85284182c7b520b'

labels_save_path = './Labels'
image_crawler_save_path = './Image_crawler'
labels_path = os.path.join(labels_save_path, 'labels.csv')
labels_dict_path = os.path.join(labels_save_path, 'labels_dict.npy')
flickr_scrape_save_path = os.path.join(image_crawler_save_path, 'Flickr_scrape')

if not os.path.exists(labels_path):
    print("{} didn't exist!".format(labels_path))
else:
    labels_pd = pd.read_csv(labels_path, names=['trash_type', 'item_type'])

# print(labels_pd)

if len(sys.argv) != 1:
    type_numbers = int(sys.argv[1])
    # labels_list = labels_pd.sample(n=5).values.tolist()
    labels_list = labels_pd.groupby(labels_pd['trash_type']).apply(lambda x:x.sample(n=random.randint(1,type_numbers))).values.tolist()

else:
    labels_list = []
    labels_list.extend(labels_pd.loc[labels_pd['item_type'].isin(['Pen', 'Vegetables', 'Battery', 'Newspaper', 'Pillow'])].values.tolist())

print("We plan to use these labels: {}".format(labels_list))
labels_dict = dict()
for row in labels_list:
    labels_dict[row[-1]] = '_'.join(row)
    
print("We plan to use these labels: {}".format(labels_dict))

# Remove past data
def remove_past_scrape(flickr_scrape_save_path):
    flickr_scrape_save_path = os.path.join(image_crawler_save_path, 'Flickr_scrape')
    try:
        shutil.rmtree(flickr_scrape_save_path)
        print(f"Removed past scrape in {image_crawler_save_path}")
    except OSError as e:
        print("Error: %s : %s" % (flickr_scrape_save_path, e.strerror))


remove_past_scrape(flickr_scrape_save_path)
# create a new folder
if not os.path.exists(flickr_scrape_save_path):
    os.makedirs(flickr_scrape_save_path)
    print('We create a new folder {}'.format(flickr_scrape_save_path))
    
def fetch_image_link(query):
    flickr = FlickrAPI(key, secret)        #initialize python flickr api
    photos = flickr.walk(text=query,
                tag_mode='all',
                extras='url_c',      #specify meta data to be fetched
                sort='relevance')     #sort search result based on relevance (high to low by default)
    
    if len(sys.argv) > 2:
        max_count = int(sys.argv[2])
    else:
        max_count = 100                 #number of images
        
    urls = []
    count = 0

    for photo in photos:
        if count < max_count:
            count = count + 1
            #print("Fetching url for image number {}".format(count))
            try:
                url = photo.get('url_c')
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
        else:
            print(f"Done fetching {query} urls, fetched {len(urls)} urls out of {max_count}")
            break
    return urls


#　Delete the labels that cannot get the image URL
# Otherwise, save URL into ./Image_crawler/Flickr_scrape
labels_dict_copy = labels_dict.copy()
for query, value in labels_dict.items():
    urls = fetch_image_link(query)
    try:
        print('example url:', urls[0])
        urls = pd.Series(urls)
        category_path = f'{flickr_scrape_save_path}/{value}_urls.csv'
        print(f"Writing {query} urls to {category_path}")
        urls.to_csv(category_path)
    except:
        labels_dict_copy.pop(query)
labels_dict = labels_dict_copy

np.save(labels_dict_path, labels_dict)



def fetch_files_with_link(url_path, item_type):
    with open(url_path, newline="") as csvfile:
        urls = pd.read_csv(url_path, delimiter=',')
        urls = urls.iloc[:, 1].to_dict().values()
        
    SAVE_PATH = os.path.join(url_path.replace('_urls.csv', ''))
    proxies = {'http': 'http://45.152.188.236:3128', 
        #    'https': 'https://169.57.1.85:8123',
          }
        #   {'http': '181.176.211.168'}
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH) #define image storage path
    
    count = 0
    for idx, url in tqdm(enumerate(urls), total=len(urls)):
        # print(url)
        # print("Starting download {} of ".format(url[0] + 1), len(urls))
        try:
            resp = requests.get(url, proxies=proxies, stream=True)   #request file using url
            # resp = requests.get(url, stream=True)   #request file using url
            count += 1
            picture_type = url.split("/")[-1].split('.')[-1]
            path_to_write = os.path.join(SAVE_PATH, item_type + '_' + str(count) + '.' + picture_type)
            # print(path_to_write)
            outfile = open(path_to_write, 'wb')
            outfile.write(resp.content) #save file content
            outfile.close()
            # print("Done downloading {} of {}".format(idx + 1, len(urls)))
        except:
            print("Failed to download url number {}".format(idx))
        finally:
            time.sleep(0.5)
    print(f"Done with {url_path} download, images are saved in {SAVE_PATH}")


print("Start downloading images...")

for label, garbage_type in labels_dict.items():
    url_path = f'{flickr_scrape_save_path}/{garbage_type}_urls.csv'
    # print(url_path)
    fetch_files_with_link(url_path, label)