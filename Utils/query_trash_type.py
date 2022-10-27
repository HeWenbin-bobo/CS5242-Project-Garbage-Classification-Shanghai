from bs4 import BeautifulSoup
import time
import http.client
import hashlib
import urllib
import random
import json
from tqdm import tqdm
import os

labels_save_path = './Labels'

if not os.path.exists(labels_save_path):
    os.makedirs(labels_save_path)
    print("We create a new directory ./Labels")
    
def get_garbage_type(labels_save_path):
    base_url = 'http://trash.lhsr.cn/sites/feiguan/trashTypes_3/TrashQuery.aspx?'

    labels = dict()

    # For those labels not included in Shanghai system, their types are marked as 'N/A'
    # Sometimes, we may not able to get the type from the system due to unexpected situations, such as network failure
    with open(os.path.join(labels_save_path, 'labels.txt'), 'r', encoding='utf-8') as f:
        for l in f:
            if len(l)<=4:
                labels[l.strip()] = 'N/A'

    # Use the proxy to avoid our IP address being blocked       
    proxy = {'http': '181.176.211.168'}
    proxy_handler = urllib.request.ProxyHandler(proxy)
    opener = urllib.request.build_opener(proxy_handler)
    urllib.request.install_opener(opener)

    count = 0
    for idx, label in tqdm(enumerate(labels.keys()), total=len(labels.keys())):
        count += 1
        # if count > 10:break
        url = base_url + urllib.parse.urlencode({'kw': label}) # Send the label as 'kw' to Shanghai system to get reply
        try:
            with urllib.request.urlopen(url) as f:
                res = f.read().decode('utf-8')
                soup = BeautifulSoup(res, 'html.parser')
        except:
            print(label, 'Error') # If Shanghai system doesn't reply, then print warning message
        else:
            type_info = soup.find_all('script', {'type': 'text/javascript'}) # Get the actual type information under <script ...>
            if type_info is not None: # If Shanghai system can classify this label or find some similar labels, then go to next step
                type_info = str(type_info[-1].string).split("'") # Get the innerText in the response
                # print(type_info)
                if label == type_info[1]: # Labels is on the second position, while garbage type is on the fourth position
                    garbage_type = type_info[3]
                else:
                    garbage_type = 'N/A' # If the system return a wrong label (or it only returns some similar labels), just set its type as 'N/A'
            else:
                garbage_type = 'N/A'

            # print(label, '->', garbage_type)
            labels[label] = garbage_type

            time.sleep(1)
    
    return labels


print("Start querying garbage type from http://trash.lhsr.cn/sites/feiguan/trashTypes_3/TrashQuery.aspx?...")
labels = get_garbage_type(labels_save_path)




appid = '20220929001359184'         # APPID
secretKey = '0I82ZekjAUqQ4A1vbozJ'      # KEY

def baidu_translate(appid, secretKey, labels):
    labels_translated = dict()

    httpClient = None
    myurl = '/api/trans/vip/translate'      # API link
    
    fromLang = 'auto'              # Original language
    toLang = 'en'                # Translation language
    salt = random.randint(32768, 65536)

    count = 0
    for idx, (label, garbage_type) in tqdm(enumerate(labels.items()), total=len(labels.items())):
        count += 1
        # if count > 10:break
        sign = appid + label + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl_label = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(label) + '&from=' + fromLang + \
                '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
            
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl_label)
            response = httpClient.getresponse()
            translated_result = response.read().decode("utf-8")
            translated_result = json.loads(translated_result)
            # print(translated_result)
            label_translated = translated_result['trans_result'][0]['dst'].capitalize()
            
            if garbage_type != 'N/A':
                sign = appid + garbage_type + str(salt) + secretKey
                sign = hashlib.md5(sign.encode()).hexdigest()
                myurl_garbage_type = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(garbage_type) + '&from=' + fromLang + \
                            '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
                httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
                httpClient.request('GET', myurl_garbage_type)
                response = httpClient.getresponse()
                translated_result = response.read().decode("utf-8")
                translated_result = json.loads(translated_result)
                # print(translated_result)
                garbage_type_translated = translated_result['trans_result'][0]['dst'].capitalize()
                labels_translated[label_translated] = garbage_type_translated
                # print(label, '->', label_translated, ',', garbage_type, '->', garbage_type_translated)
            else:
                labels_translated[label_translated] = garbage_type
            
        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()

            time.sleep(1)
        
    return labels_translated


print("Start translating labels through Baidu API...")
labels_translated = baidu_translate(appid, secretKey, labels)




def save_labels_csv(labels_translated, labels_save_path):
    with open(os.path.join(labels_save_path, 'labels.csv'), 'w', encoding='utf-8') as f:
        for label, garbage_type in labels_translated.items():
            if garbage_type != 'N/A' and len(label.split()) <= 1: # If the translation for a label contains over three words, it may be hard to get the images later
                f.write(','.join([garbage_type, label]))
                f.write('\n')


print("Start saving labels into csv...")
save_labels_csv(labels_translated, labels_save_path)
print("Finish saving labels into csv")    