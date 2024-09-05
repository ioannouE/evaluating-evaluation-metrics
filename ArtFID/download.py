import pandas as pd
import os
from tqdm import tqdm
from urllib.request import urlretrieve
import urllib
import requests

# opener = urllib.request.build_opener()
# header = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
# opener.addheaders = [('User-Agent', header)]
# urllib.request.install_opener(opener)


def get_pics(urls, artists, styles):
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)
    for idx, url in enumerate(tqdm(urls)):
        output_path = os.path.join(output_dir, str(styles[idx]).replace(" ","")+'_'+str(artists[idx]).replace(" ","")+'.jpg')
        try:
            urlretrieve(url.strip(), output_path)
        except Exception:
            print(url)
            continue

def get_images(urls, artists, styles):
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)
    for idx, url in enumerate(tqdm(urls)):
        # img_data = requests.get(url, stream=True)#.content

        # if not img_data.ok:
        #     print(img_data)

        output_path = os.path.join(output_dir, str(styles[idx]).strip()+'_'+str(idx+217514) + '_' + str(artists[idx]).strip().replace(' ','').replace('/','').replace('.', '').replace(',','')+'.jpg')
        with open(output_path, 'wb') as handler:
            response = requests.get(url.strip(), stream=True)

            if not response.ok:
                print(response, url)

            for block in response.iter_content(1024):
                if not block:
                    break
                handler.write(block)


if __name__ == '__main__':
    df = pd.read_csv('artfid_dataset_copy.csv')
    urls = df['url'].values.tolist()
    # names = df['image_name'].values.tolist()
    artists = df['artist'].values.tolist()
    styles = df['style'].values.tolist()

    # get_pics(urls, artists, styles)
    get_images(urls, artists, styles)

