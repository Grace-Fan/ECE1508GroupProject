import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir.parent / 'dataset'
model_dir = current_dir.parent / 'model'

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path, total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    flickr_directory=dataset_dir/'Flickr8k'
    os.makedirs(flickr_directory, exist_ok=True)

    # URLs for Flickr8k dataset (hosted on University of Illinois or Kaggle alternative)
    images_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
    captions_url = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'

    images_zip = os.path.join(flickr_directory, 'Flickr8k_Dataset.zip')
    captions_zip = os.path.join(flickr_directory, 'Flickr8k_text.zip')

    print("Downloading Flickr8k images...")
    if not os.path.exists(images_zip):
        download_file(images_url, images_zip)
    else:
        print("Images zip already downloaded.")

    print("Downloading Flickr8k captions...")
    if not os.path.exists(captions_zip):
        download_file(captions_url, captions_zip)
    else:
        print("Captions zip already downloaded.")

    print("Extracting images...")
    extract_zip(images_zip, os.path.join(flickr_directory, 'Images'))

    print("Extracting captions...")
    extract_zip(captions_zip, os.path.join(flickr_directory, 'Captions'))

    print("Download and extraction completed.")

if __name__ == '__main__':
    main()
