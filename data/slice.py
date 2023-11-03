import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_beats_aistpp(motion_dir, wav_dir, beats_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    beats = sorted(glob.glob(f"{beats_dir}/*.txt"))
    sliced_dir_str = "_beats_sliced"
    wav_out = wav_dir + sliced_dir_str
    motion_out = motion_dir + sliced_dir_str
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    # assert len(wavs) == len(motions)
    
    for idx in tqdm(range(len(beats))):
        beat = beats[idx]
        f_id = "".join(os.path.basename(beat).split(".")[:-1])
        motion = os.path.join(motion_dir, f_id + ".pkl")
        wav = os.path.join(wav_dir, f_id + ".wav")

        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        b_name = os.path.splitext(os.path.basename(beat))[0]
        assert m_name == w_name and m_name == b_name, str((motion, wav, beat))

        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_beats_n_motion(motion, beat, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_beats_n_motion(motion_file, beat_file, stride, length, num_slices, out_dir):
    def read_beat_file(beat_file):
        with open(beat_file) as f:
            beat_ids = f.readline()
        beat_ids = beat_ids.split(",")
        beat_ids = [int(v) for v in beat_ids]
        return beat_ids 
 
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    # read beat frame idx (starting from 1)
    # from annitation file
    beat_ids = read_beat_file(beat_file)
    beat_ids = np.array(beat_ids)
    
    # prepare for downsampling by 2
    beat_mask = np.zeros((len(pos),), dtype="int")
    if beat_ids[-1] >= len(pos):
        beat_ids[-1] -= 1
    beat_mask[beat_ids - 1] = 1
    beat_mask[beat_ids] = 1

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice, b_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
            beat_mask[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice, "beat": b_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
