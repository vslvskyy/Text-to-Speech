import os
import torch
import librosa

from tqdm import tqdm
from configs import TrainConfig


def get_target_energy(train_config: TrainConfig):
    wavs_paths = sorted([
        os.path.join(train_config.wavs_path, f) for f in os.listdir(train_config.wavs_path)
                                                    if os.path.isfile(os.path.join(train_config.wavs_path, f))
    ])

    energies = {}
    for wav_path in tqdm(wavs_paths, total=len(wavs_paths)):
        wav_tensor, sr = librosa.load(wav_path)

        energy = torch.stft(
            torch.tensor(wav_tensor).to(train_config.device),
            n_fft=train_config.n_fft,
            hop_length=train_config.hop_length
        ).transpose(0, 1)

        energy = torch.sqrt(energy[:, :, 0]**2 + energy[:, :, 1]**2)
        energy = torch.linalg.norm(energy, dim=-1)

        energies[wav_path] = energy

    return energies
