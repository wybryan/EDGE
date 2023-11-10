import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.baseline_features import beat_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    stride_length = opt.stride
    sample_length = opt.out_length
    sample_size = int(sample_length / stride_length) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    all_beat_feat = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, max(0, len(file_list) - sample_size))
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + min(len(file_list), sample_size)]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, stride_length, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, max(0, len(file_list) - sample_size))
            cond_list = []
            beat_list = []

            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                beat_per_file = beat_extract(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
                    beat_list.append(beat_per_file)
            cond_list = torch.from_numpy(np.array(cond_list))
            beat_list = torch.from_numpy(np.array(beat_list)).to(torch.int64)
            all_beat_feat.append(beat_list)
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + min(len(file_list), sample_size)])

    model = EDGE(opt.feature_type, opt.checkpoint, EMA=False, use_music_beat_feat=opt.use_music_beat_feat)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir
        os.makedirs(fk_out, exist_ok=True)

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        beat_feat = all_beat_feat[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, beat_feat, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    opt.out_length = 10
    opt.no_render = True
    opt.use_music_beat_feat = True
    exp_name = "exp29"
    for i in range(1):
        epoch_no = i + 1
        opt.motion_save_dir = f"eval/{exp_name}/beats_on_motion_{epoch_no}e"
        opt.render_dir = opt.motion_save_dir
        opt.checkpoint = f"/Projects/Github/paper_project/EDGE/runs/train/{exp_name}/weights/train-{epoch_no}.pt"
        # model = EDGE(opt.feature_type, opt.checkpoint, EMA=False, use_music_beat_feat=opt.use_music_beat_feat)
        test(opt)
