import torch
import numpy as np

from utils import text_to_sequence


def synthesize(model, text, device, length_coef=1.0, pitch_coef=1.0, energy_coef=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(
            sequence,
            src_pos,
            length_coef=length_coef,
            pitch_coef=pitch_coef,
            energy_coef=energy_coef
    )
    return mel.contiguous().transpose(1, 2)


def get_data(tests, train_config):
    data_list = list(text_to_sequence(test, train_config.text_cleaners) for test in tests)
    return data_list

