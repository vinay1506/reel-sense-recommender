
import os
import zipfile
import requests
from tqdm import tqdm

def download_movielens_dataset(url, save_path):
    """
    Download the MovieLens dataset
    
    Parameters:
    url (str): URL to download the dataset from
    save_path (str): Directory to save the dataset to
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "dataset.zip")
    
    # Download the file
    print(f"Downloading MovieLens dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(file_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    # Extract the dataset
    print(f"Extracting dataset to {save_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    
    # Remove the zip file
    os.remove(file_path)
    
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    # MovieLens-Small (100K) dataset
    small_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    download_movielens_dataset(small_url, "ml-latest-small")
    
    # Uncomment to download the full dataset (25M)
    # full_url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    # download_movielens_dataset(full_url, "ml-latest")
