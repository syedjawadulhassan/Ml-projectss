"""
Download and set up the Chest X-Ray Pneumonia dataset.
Creates: dataset/train/NORMAL, dataset/train/PNEUMONIA, dataset/val/..., dataset/test/...

Tries in order:
  1. Kaggle API (if kaggle is installed and API key is set)
  2. GitHub mirror (no account needed)
"""

import os
import zipfile
import shutil
import sys
import urllib.request

# GitHub mirror (same data as Kaggle, from probml/chest_xray_kaggle)
GITHUB_ZIP = "https://github.com/probml/chest_xray_kaggle/archive/refs/heads/main.zip"


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_kaggle(root):
    """Install kaggle into project folder if not available (avoids system Temp)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi
    except ImportError:
        pass
    lib_dir = os.path.join(root, "kaggle_lib")
    os.makedirs(lib_dir, exist_ok=True)
    print("Installing kaggle into project...")
    import subprocess
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", lib_dir,
            "kaggle",
        ],
        check=True,
        capture_output=True,
    )
    sys.path.insert(0, lib_dir)
    from kaggle.api.kaggle_api_extended import KaggleApi
    return KaggleApi


def download_via_kaggle(download_dir, root):
    api_class = _ensure_kaggle(root)
    api = api_class()
    api.authenticate()
    api.dataset_download_files(
        "paultimothymooney/chest-xray-pneumonia",
        path=download_dir,
        unzip=True,
    )
    return download_dir


def download_via_github(download_dir):
    """Download dataset from GitHub mirror (no API key needed)."""
    zip_path = os.path.join(download_dir, "chest_xray_kaggle.zip")
    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print("\r  Progress: %d%%" % pct, end="", flush=True)
    try:
        urllib.request.urlretrieve(GITHUB_ZIP, zip_path, reporthook=_progress)
    finally:
        print()
    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    # Repo extracts to chest_xray_kaggle-main/
    return os.path.join(extract_dir, "chest_xray_kaggle-main")


def find_split(base, name):
    for dirpath, dirnames, _ in os.walk(base):
        if os.path.basename(dirpath) == name:
            return dirpath
        for d in dirnames:
            if d == name:
                return os.path.join(dirpath, d)
    return None


def organize_into_dataset(extract_dir, dataset_dir):
    train_src = find_split(extract_dir, "train")
    val_src = find_split(extract_dir, "val")
    test_src = find_split(extract_dir, "test")

    if not train_src or not os.path.isdir(train_src):
        print("Extracted structure:")
        for r, dirs, _ in os.walk(extract_dir):
            level = r[len(extract_dir) :].count(os.sep)
            print("  " * level + os.path.basename(r) + "/")
        return False

    for split_name, src in [("train", train_src), ("val", val_src), ("test", test_src)]:
        if not src or not os.path.isdir(src):
            print("Warning: %s not found, skipping." % split_name)
            continue
        dest_split = os.path.join(dataset_dir, split_name)
        os.makedirs(dest_split, exist_ok=True)
        for class_name in ["NORMAL", "PNEUMONIA"]:
            src_class = os.path.join(src, class_name)
            dest_class = os.path.join(dest_split, class_name)
            if os.path.isdir(src_class):
                if os.path.isdir(dest_class):
                    shutil.rmtree(dest_class)
                shutil.copytree(src_class, dest_class)
                n = len(
                    [
                        f
                        for f in os.listdir(dest_class)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )
                print("  %s/%s: %d images" % (split_name, class_name, n))
            else:
                os.makedirs(dest_class, exist_ok=True)
                print("  %s/%s: (empty)" % (split_name, class_name))
    return True


def main():
    root = get_project_root()
    dataset_dir = os.path.join(root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    download_dir = os.path.join(root, "dataset_download")
    os.makedirs(download_dir, exist_ok=True)

    extract_dir = None

    # Prefer GitHub (no account or pip install needed)
    try:
        print("Downloading from GitHub mirror (no account required)...", flush=True)
        extract_dir = download_via_github(download_dir)
    except Exception as e:
        print("GitHub download failed:", e, flush=True)
        extract_dir = None

    # Fallback: Kaggle (requires pip install kaggle and API key)
    if extract_dir is None or not os.path.isdir(extract_dir) or not find_split(extract_dir, "train"):
        if os.path.isdir(download_dir):
            shutil.rmtree(download_dir, ignore_errors=True)
        os.makedirs(download_dir, exist_ok=True)
        try:
            print("Trying Kaggle...", flush=True)
            download_via_kaggle(download_dir, root)
            extract_dir = download_dir
            for name in ["chest_xray", "chest-xray-pneumonia", "Chest_X_Ray_Images_Pneumonia"]:
                candidate = os.path.join(download_dir, name)
                if os.path.isdir(candidate):
                    extract_dir = candidate
                    break
        except Exception as e:
            print("Kaggle failed:", e, flush=True)
            print("Check your internet connection, or set up Kaggle API key.", flush=True)
            sys.exit(1)

    if not organize_into_dataset(extract_dir, dataset_dir):
        print("Could not find train/val/test in downloaded data.")
        sys.exit(1)

    print("Cleaning up...")
    shutil.rmtree(download_dir, ignore_errors=True)
    print("Done. Dataset at:", os.path.abspath(dataset_dir))


if __name__ == "__main__":
    main()
