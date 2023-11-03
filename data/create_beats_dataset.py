import argparse
import os
from pathlib import Path
import shutil

from filter_split_data import *
from slice import *


def copy_music_feats(search_dir, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    fpaths = Path(search_dir).glob("*")
    for fpath in fpaths:
        audio_name = Path(fpath).stem
        save_path = os.path.join(dest_dir, audio_name + ".npy")
        source_f = os.path.join(source_dir, os.path.basename(save_path))
        shutil.copy(source_f, save_path)


def create_dataset(opt):
    # split the data according to the splits files
    # print("Creating train / test split")
    # split_data(opt.dataset_folder)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_beats_aistpp(f"train/motions", f"train/wavs", f"train/beats", opt.stride, opt.length)

    # copy music features
    copy_music_feats("train/wavs_sliced", "train_0/baseline_feats", "train/baseline_feats")
    copy_music_feats("train/wavs_sliced", "train_0/jukebox_feats", "train/jukebox_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5, help="sliced data overlap length in seconds")
    parser.add_argument("--length", type=float, default=5.0, help="sliced data length in seconds")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
