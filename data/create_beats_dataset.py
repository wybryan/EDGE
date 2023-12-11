import argparse
import os
from pathlib import Path
import shutil

from filter_split_data import *
from slice import *

from audio_extraction.BEATs_features import extract_folder as BEATs_extract

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
    slice_beats_aistpp(f"train/motions", f"train/wavs", f"train/beats", opt.stride, opt.length, beat_file_limit=opt.dataset_size)

    # copy music features
    copy_music_feats("train/wavs_sliced", "train_full/baseline_feats", "train/baseline_feats")
    copy_music_feats("train/wavs_sliced", "train_full/jukebox_feats", "train/jukebox_feats")

    # generate features based on BEATs model
    BEATs_extract(opt.beats_model_path, "train/wavs_sliced", "train/BEATs_feats")
    BEATs_extract(opt.beats_model_path, "test/wavs_sliced", "test/BEATs_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5, help="sliced data overlap length in seconds")
    parser.add_argument("--length", type=float, default=5.0, help="sliced data length in seconds")
    parser.add_argument("--dataset_size", type=int, default=-1, help="# beat file limit")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument(
        "--beats_model_path",
        type=str,
        default='audio_extraction/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
        help="BEATs model weights",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # opt.dataset_size = 20
    # opt.beats_model_path = 'audio_extraction/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    create_dataset(opt)
