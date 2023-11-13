import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import pickle

import numpy as np
import torch
from tqdm import tqdm

from p_tqdm import p_map
from vis import visualize_data

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.BEATs_features import extract as beat_extract, init_model
from data.audio_extraction.jukebox_features import extract as juke_extract

from accelerate.utils import set_seed

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

def read_pkl(fname):
    with open(fname, "rb") as f:
        meta_data = pickle.load(f)
    return meta_data

def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    stride_length = opt.stride
    sample_length = opt.out_length
    sample_size = int(sample_length / stride_length) - 1
    rand_idx = 0

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
            juke_file_list = sorted(glob.glob(f"{dir}/*.pkl"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            # rand_idx = random.randint(0, max(0, len(file_list) - sample_size))
            
            file_list = file_list[rand_idx : rand_idx + min(len(file_list), sample_size)]
            juke_file_list = juke_file_list[rand_idx : rand_idx + min(len(file_list), sample_size)]
            
            meta_data_dicts = [read_pkl(x) for x in juke_file_list]
            cond_list = [x["music_feat"] for x in meta_data_dicts]
            beat_list = [x["beat_onehot"] for x in meta_data_dicts]

            cond_list = torch.from_numpy(np.array(cond_list))
            beat_list = torch.from_numpy(np.array(beat_list)).to(torch.int64)

            all_cond.append(cond_list)
            all_beat_feat.append(beat_list)
            all_filenames.append(file_list)
    else:
        print("Computing features for input music")
        BEATs_model = init_model(opt.beats_model_path)
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
            # rand_idx = random.randint(0, max(0, len(file_list) - sample_size))
            
            cond_list = []
            beat_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    print(f"skipping {file} with index = {idx}...")
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                beat_per_file, _ = beat_extract(file, BEATs_model)
                # save reps
                if opt.cache_features:
                    meta_data = dict(music_feat=reps, beat_onehot=beat_per_file)
                    featurename = os.path.splitext(file)[0] + ".pkl"
                    with open(featurename, "wb") as f:
                        pickle.dump(meta_data, f)

                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
                    beat_list.append(beat_per_file)

            cond_list = torch.from_numpy(np.array(cond_list))
            beat_list = torch.from_numpy(np.array(beat_list))
            file_list = file_list[rand_idx : rand_idx + min(len(file_list), sample_size)]

            all_beat_feat.append(beat_list)
            all_cond.append(cond_list)
            all_filenames.append(file_list)

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
    set_seed(42)
    opt = parse_test_opt()
    # opt.cache_features = True
    opt.use_cached_features = True
    opt.out_length = 10
    opt.no_render = True
    opt.use_music_beat_feat = True

    exp_name = "exp71"
    for i in range(1):
        epoch_no = i + 1
        opt.motion_save_dir = f"eval/{exp_name}/beats_on_motion_{epoch_no}e"
        opt.render_dir = opt.motion_save_dir
        opt.checkpoint = f"/Projects/Github/paper_project/EDGE/runs/train/{exp_name}/weights/train-{epoch_no}.pt"
        test(opt)
    
    # gen visual    
    motions_all = []
    out_dir_all = []
    for i in range(1):
        epoch_no = i + 1
        motion_dir = f"/Projects/Github/paper_project/EDGE/eval/{exp_name}/beats_on_motion_{epoch_no}e"
        motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
        out_dir = motion_dir
        out_dir = [out_dir for _ in range(len(motions))]
        out_dir_all = out_dir_all + out_dir
        motions_all = motions_all + motions

    def inner(x):
        motion_file, out_dir = x
        visualize_data(motion_file, render_out_dir=out_dir, render_gif_fname=os.path.basename(motion_file) + ".gif")
    p_map(inner, zip(motions_all, out_dir_all))
    
    # # baseline
    # opt.motion_save_dir = f"/Projects/Github/paper_project/EDGE/eval/beats_off_motion"
    # opt.render_dir = opt.motion_save_dir
    # opt.checkpoint = f"/Projects/Github/paper_project/EDGE/checkpoint.pt"
    # test(opt)

