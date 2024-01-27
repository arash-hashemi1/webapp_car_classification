from data_cleaning import clean_data
from random import uniform
import pandas as pd
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO
from string import ascii_letters
from numpy.random import choice
from pathlib import Path
from time import sleep
import random



def random_sleep(min_time: float = 0.4, max_time: float = 0.8):
  sleep_time = uniform(min_time, max_time)
  sleep(sleep_time)

def get_pic(link:str, label:str, image_dir:str, idx:int):

  headers={'User-Agent': 'Opera/9.80 (X11; Linux i686; Ub'
          'untu/14.10) Presto/2.12.388 Version/12.16'}

  try:
    random_sleep()
    print(link)
    r = requests.get(link, timeout=10, headers=headers)
    im = Image.open(BytesIO(r.content))
    folder_name = os.path.join(image_dir, label)
    if os.path.exists(folder_name) == False:
      os.mkdir(folder_name)
    im_name = f'{label}_{idx}.jpg'
    file_name = os.path.join(folder_name, im_name)
    im.save(file_name)
  except:
    print(f'Problem with {link}')


def save_images(df:pd.DataFrame, image_dir:str):
  for label in df.index:
    url_list = df.loc[label].dropna()
    for idx, url in enumerate(url_list, 1):
       get_pic(url, label, image_dir, idx)


if __name__ == '__main__':
  data_dir = '/content/drive/MyDrive/CarClassificationProject/data_scrapping'
  image_dir = '/content/drive/MyDrive/CarClassificationProject/image_full'
  clean_data(data_dir = '/content/drive/MyDrive/CarClassificationProject/data_scrapping')
  df_pic_url = pd.read_csv(os.path.join(data_dir, 'pic-url.csv'), index_col=0)
  save_images(df_pic_url, image_dir)