"""
Sources

https://www.geeksforgeeks.org/python/python-pillow-tutorial/
https://www.geeksforgeeks.org/python/asyncio-in-python/
https://www.geeksforgeeks.org/python/how-to-download-an-image-from-a-url-in-python/

AI Usage:
- AI was used to generate much of the scaffholding for this code, to enable me to efficiently
write this code to download the images
    - https://sandbox.ai.huit.harvard.edu/share/v1BosMUiEzJFsmOH8BPRd
- AI was used to debug issues of URLs giving error 406. Here are some AI chat URLs:
    - https://sandbox.ai.huit.harvard.edu/share/-WTeJNw2mRSPvUy_dSPGO
- AI was used to fill in missing alphanumeric characters in some image URLs
- AI was used to understand how to download datasets from HuggingFace and get specific images from them
(to fill in missing images from SkinCAP)

"""

import pandas as pd
from pathlib import Path
import asyncio
import aiohttp
import aiofiles
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import shutil

HF_SECRET_PATH = "/Users/nicholasyousefi/Documents/Coding/School/Harvard/Fall_2025/ac215/project/other_secrets/huggingface_read_repos.txt"
load_dotenv(HF_SECRET_PATH)

REPOROOT = Path("..").resolve()
CSV_PATH = REPOROOT / 'data' / 'fitzpatrick17k.csv'
SKINCAP_CSV_PATH = REPOROOT / 'data' / 'skincap_v240623.csv'

OUTPUT_DIR = REPOROOT / 'data_preprocessed'
OUTPUT_DIR.mkdir(exist_ok=True)

# Limit concurrent connections so you don't hammer servers / your network
MAX_CONCURRENT = 20

FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
}

def read_urls_from_csv(csv_path):
    csv_df = pd.read_csv(csv_path)

    # TEMPORARY: subset csv_df for testing
    # csv_df = csv_df.iloc[0:10, :]

    urls = csv_df['url'].to_list()
    hashes = csv_df['md5hash'].to_list()
    return urls, hashes


def generate_filename(hsh, ext):
    """
    Generate a filename from hash and extension.
    """
    return hsh + ext

async def fetch_and_save(session, url, hsh, sem):
    async with sem:  # limit concurrency
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    error = {
                        'hash': hsh,
                        'url': url,
                        'error': resp.status,
                    }
                    return error
                data = await resp.read()

            # detect format
            img = Image.open(BytesIO(data))
            fmt = img.format  # e.g. "JPEG", "PNG"
            ext = FORMAT_TO_EXT.get(fmt, ".bin")

            filename = generate_filename(hsh, ext)
            out_path = OUTPUT_DIR / filename

            async with aiofiles.open(out_path, "wb") as f:
                await f.write(data)

            return None
        except Exception as e:
            error = {
                        'hash': hsh,
                        'url': url,
                        'error': e,
                    }
            return error
        
async def fetch_skincap_image(
        img_name: str,
        hash: str,
        image_path_hf: str = "skincap/{filename}",
        dest_path: str | Path = OUTPUT_DIR
    ):
    """
    Fetch a SkinCAP image

    :param img_name: The name of the image file in SkinCAP
    :param hash: The hash of the image in FitzPatrick
    :param image_path_hf: The path to the image in the HuggingFace dataset
    :param dest_path: The path to where the destination image should be.
    """

    try:
        local_path = hf_hub_download(
            repo_id="joshuachou/SkinCAP",
            filename=image_path_hf.format(filename=img_name),
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),  # or omit if public and you’re logged in via cache
        )

        extension = Path(local_path).suffix
        filename = str(hash) + str(extension)
        dest = shutil.copy(local_path, Path(OUTPUT_DIR) / dest_path, follow_symlinks=True)
        shutil.move(dest, Path(OUTPUT_DIR) / dest_path / filename)

        return None
    except Exception as e:
        error = {
                    'hash': hash,
                    'error': e,
                }
        return error

        
async def download_missed_images_from_hf(failure_results: list[dict[str, str]], skincap_meta: pd.DataFrame):
    """
    Download any images that could not be downloaded from the Fitzpatrick using SkinCAP

    :param failure_results: A list of dictionaries containing:
    ```python
    {
        'hash': <hash>,
        'url': <url>,
        'error': <error>
    }
    :param skincap_meta: The metadata for the SkinCAP dataset
    ```
    """
    new_failures = []

    # drop Nones if needed
    failure_results_clean = [r for r in failure_results if r is not None]

    skincap_meta_prep = skincap_meta.set_index('hash').copy()

    hashes = [i['hash'] for i in failure_results_clean]
    hashes_in_skincap = pd.Series(hashes).isin(skincap_meta_prep.index).to_list()
    new_failures = new_failures + [
        {
            'hash': hashes[i],
            'error': 'Not in SkinCAP'
        } for i in range(len(hashes)) if not hashes_in_skincap[i]
    ]
    filenames = skincap_meta_prep.loc[[hashes[i] for i in range(len(hashes)) if hashes_in_skincap[i]], 'skincap_file_path'].to_list()

    tasks = [
        asyncio.create_task(fetch_skincap_image(filename, hash))
        for hash, filename in zip(hashes, filenames)
    ]
    new_failures = new_failures + await tqdm_asyncio.gather(*tasks)

    return new_failures


def read_skincap_csv(path: str | Path):
    # load the csv
    csv = pd.read_csv(path)

    # only keep relevant columns
    simplified = csv.loc[
        csv['source'] == 'fitzpatrick17k',
        ['id', 'skincap_file_path', 'ori_file_path']
    ]
    # add hash column
    simplified['hash'] = simplified['ori_file_path'].apply(lambda x: Path(x).stem)

    return simplified


async def main_save():
    print("Reading csvs")
    urls, hashes = read_urls_from_csv(CSV_PATH)
    skincap_csv = read_skincap_csv(SKINCAP_CSV_PATH)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    print("Downloading images")
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(fetch_and_save(session, url, hsh, sem))
            for url, hsh in zip(urls, hashes)
        ]
        results = await tqdm_asyncio.gather(*tasks)

    # drop Nones
    results_clean = [r for r in results if r is not None]

    # save fitzpatrick error file
    print("Saving error file")
    error_df = pd.DataFrame(results_clean)
    error_df.to_csv(OUTPUT_DIR / 'fitzpatrick_failed.csv', index=False)


    print("Downloading missed images")
    skincap_failures = await download_missed_images_from_hf(results_clean, skincap_csv)
    skincap_failures_clean = [r for r in skincap_failures if r is not None]

    error_df_skincap = pd.DataFrame(skincap_failures_clean)
    error_df_skincap.to_csv(OUTPUT_DIR / 'skincap_failed.csv', index=False)
    

if __name__ == "__main__":
    asyncio.run(main_save())
