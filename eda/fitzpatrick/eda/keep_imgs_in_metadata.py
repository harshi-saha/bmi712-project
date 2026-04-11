DOCS = """
A python script for the Fitzpatrick dataset to keep images we are using, and discard images we are
not using.

This program gives the user the option to either delete the images we are not using from a target folder
or to symlink the images in a source folder into a destination folder. Symlink is used to not duplicate
the images. The images can be duplicated using the `cp` command with the -L flag, or it can be directly
zipped into a zip file using the `zip` command (symlinks will be followed automatically).
"""
import argparse
import pandas as pd
from pathlib import Path

IMG_EXT = [
    ".jpg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
]

def keep_images_in_metadata(
        metadata_path, 
        image_dir, 
        symlink_dir=None,
        filename_col=None,
        surpress_delete_warning=False,
    ):
    # load metadata
    metadata = pd.read_csv(metadata_path)
    filenames = metadata[filename_col].to_list()

    # validate inputs
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError("img_dir must be a directory")
    
    if symlink_dir is not None:
        symlink_dir = Path(symlink_dir)
        if not symlink_dir.is_dir():
            raise ValueError("symlink-dir does not exist or is not a directory")
        
    # list images in image_dir
    image_dir_files = [entry for entry in image_dir.iterdir() if entry.is_file()]
    print('img_dir_files:', len(image_dir_files))
    image_dir_img_files = {f.stem:f.name for f in image_dir_files if not f.name.startswith(".") and f.suffix in IMG_EXT}
    print('img_dir_img_files:', len(image_dir_img_files))

    if len(filenames) > len(image_dir_img_files):
        raise RuntimeError(f"Found {len(filenames)} filenames in metadata but only {len(image_dir_img_files)} images in img_dir.")

    if symlink_dir is not None:
        # symlink all existing dirs in symlink_dir
        for filename in filenames:
            if filename in image_dir_img_files.keys():
                new_path = symlink_dir / image_dir_img_files[filename]
                old_path = image_dir / image_dir_img_files[filename]
                new_path.symlink_to(old_path)
            else:
                raise RuntimeError(f"File {filename} not found in image_dir")
        print("Symlinks created.")
    else:
        if not set(filenames).issubset(set(image_dir_img_files.keys())):
            raise RuntimeError(f"Files in metadata not found in image_dir: {set(filenames) - set(image_dir_img_files.keys())}")
        to_rm_stems = set(image_dir_img_files.keys()) - set(filenames)

        if not surpress_delete_warning:
            response = input("WARNING: the following files will be deleted. Type 'Y' to proceed 'N' to cancel: ")
            if response.lower() != 'y':
                print("Aborted.")
                return

            # delete files
            for stem in to_rm_stems:
                file = image_dir_img_files[stem]
                file.unlink()

            print("Files deleted.")
        
    


def main():
    parser = argparse.ArgumentParser(
        prog="python keep_imgs_in_metadata.py",
        description = DOCS,
    )

    parser.add_argument('metadata_path', help="The path to the metadata file containing images to keep.")
    parser.add_argument('img_dir', help='The path to the directory in which images are stored.')
    parser.add_argument('-s', '--symlink-dir', required=False, default="", help='The path to the directory where symlinks to kept images should be stored.')
    parser.add_argument('-c' '--column', required=False, default="md5hash", help="The column in the metadata containing the image file names (without extension).")
    parser.add_argument('-y', '--yes', required=False, action='store_true', help="Surpress confirmation before deleting images.")

    args = parser.parse_args()

    keep_images_in_metadata(
        metadata_path=args.metadata_path,
        image_dir=args.img_dir,
        symlink_dir=None if args.symlink_dir == "" else args.symlink_dir,
        filename_col=args.c__column,
        surpress_delete_warning=args.yes
    )

if __name__ == "__main__":
    main()
