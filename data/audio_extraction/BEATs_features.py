import os
from pathlib import Path
from functools import partial
import numpy as np
from tqdm import tqdm
import torch
import librosa

# from .beats.Tokenizers import TokenizersConfig, Tokenizers
from .beats.BEATs import BEATs, BEATsConfig


def init_model(checkpoint_path):
    # load the fine-tuned checkpoints
    checkpoint = torch.load(checkpoint_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    return BEATs_model


def extract(fpath, BEATs_model, skip_completed=True, dest_dir="aist_BEATs_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    SR = 16000
    wav_length = 5 * SR
    audio_input_16khz, _ = librosa.load(fpath, sr=SR, res_type="fft")

    # read wav
    if len(audio_input_16khz) > wav_length:
        audio_input_16khz = audio_input_16khz[:wav_length]

    raw_wav = torch.from_numpy(audio_input_16khz).unsqueeze(0)
    audio_padding_mask = torch.zeros(raw_wav.shape).bool()

    with torch.no_grad():
        audio_embeds, _ = BEATs_model.extract_features(
            raw_wav,
            padding_mask=audio_padding_mask,
            feature_only=True,
        )
    audio_embeds = audio_embeds[0]
    audio_embeds = audio_embeds.numpy()
    return audio_embeds, save_path


def extract_folder(model_path, src, dest):
    model = init_model(model_path)
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, BEATs_model=model, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        np.save(path, rep)
