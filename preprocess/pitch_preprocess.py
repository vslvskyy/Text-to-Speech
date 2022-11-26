import os
import tqdm
import torch
import pyworld
import librosa
import numpy as np

from configs import TrainConfig


def get_target_pitch_energy(train_config: TrainConfig):
    wavs_paths = sorted([
        os.path.join(train_config.wavs_path, f) for f in os.listdir(train_config.wavs_path)
                                                    if os.path.isfile(os.path.join(train_config.wavs_path, f))
    ])

    pitches = {}
    for wav_path in tqdm(wavs_paths, total=len(wavs_paths)):
        wav_tensor, sr = librosa.load(wav_path)
        pitch, t = pyworld.dio(
            wav_tensor.astype(np.float64),
            sr,
            frame_period=train_config.hop_length * 1000 / sr
        )
        pitch = pyworld.stonemask(wav_tensor.astype(np.float64), pitch, t, sr)

        pitches[wav_path] = pitch

    return pitches
