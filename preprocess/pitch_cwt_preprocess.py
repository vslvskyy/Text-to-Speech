import torch
import numpy as np

from pycwt import wavelet
from scipy.interpolate import interp1d

from typing import Tuple, TypedDict


def get_spec_from_pitch(pitch: np.array) -> np.array:
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 4
    J = 9

    pitch_spec, *_ = wavelet.cwt(np.squeeze(pitch), dt, dj, s0, J, mother)
    pitch_spec = np.real(pitch_spec).T
    return pitch_spec


def get_pitch_from_spec(pitch_spec: torch.tensor) -> torch.tensor:
    scales = torch.tensor([0.05 * 2**i for i in range(2, 12)])
    b = (torch.arange(1, len(scales) + 1).float().to(pitch_spec.device)[None, None, :] + 2.5) ** (-2.5)
    pitch = pitch_spec * b
    return pitch.sum(-1)


def smooth_pitch_countor(pitch: np.array) -> np.array:
    voiced_frames = np.where(pitch)[0]
    voiced_pitch_values = pitch[pitch != 0.]

    if 0 not in voiced_frames:
        voiced_frames = np.concatenate([
            np.array([0]),
            voiced_frames
        ])
        voiced_pitch_values = np.concatenate([
            np.array([voiced_pitch_values[0]]),
            voiced_pitch_values
        ])

    if len(pitch) - 1 not in voiced_frames:
        voiced_frames = np.concatenate([
            voiced_frames,
            np.array([len(pitch) - 1])
        ])
        voiced_pitch_values = np.concatenate([
            voiced_pitch_values,
            np.array([voiced_pitch_values[-1]])
        ])

    interpol = interp1d(voiced_frames, voiced_pitch_values)

    return interpol(np.arange(len(pitch)))


def preprocess_pitch(pitch: np.array) -> Tuple[TypedDict[str, torch.tensor], np.float, np.float]:
    x = smooth_pitch_countor(pitch)
    x = np.log(x + 1)
    pitch_mean = x.mean()
    pitch_std = x.std()
    x = x / pitch_std.item() - pitch_mean
    pitch_spec = get_spec_from_pitch(x)
    return pitch_spec, pitch_mean, pitch_std
