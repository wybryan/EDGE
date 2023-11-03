import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    # split the data according to the splits files
    # print("Creating train / test split")
    # split_data(opt.dataset_folder)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_beats_aistpp(f"train/motions", f"train/wavs", f"train/beats", opt.stride, opt.length)


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
