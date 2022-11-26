import os
import time
from tqdm import tqdm

import torch
import numpy as np

from .text_utils import text_to_sequence
from .other_utils import (
    process_text,
    pad_1D_tensor,
    pad_2D_tensor
)


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path)
    wav_paths = sorted([
        os.path.join(train_config.wavs_path, f) for f in os.listdir(train_config.wavs_path)
                                                    if os.path.isfile(os.path.join(train_config.wavs_path, f))
    ])

    preprocessed_pitches = torch.load(train_config.pitch_ground_truth)

    energies = torch.load(train_config.energy_ground_truth)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(
            os.path.join(
                train_config.alignment_path,
                str(i)+".npy"
            )
        )
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(
                character,
                train_config.text_cleaners
            )
        )

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        (
            pitch,
            pitch_spec,
            pitch_mean,
            pitch_std
        ) = preprocessed_pitches[wav_paths[i]]
        assert len(pitch) == sum(duration)

        energy = energies[wav_paths[i]]
        assert len(energy) == sum(duration)

        buffer.append({
            "text": character,
            "duration": duration,
            "mel_target": mel_gt_target,
            "pitch": torch.tensor(pitch).to(train_config.device),
            "pitch_spec": torch.tensor(pitch_spec).to(train_config.device),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy": energy.to(train_config.device)
        })

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]

    pitch_specs = [batch[ind]["pitch_spec"] for ind in cut_list]
    pitch_stats = [
        torch.tensor([
            batch[ind]["pitch_mean"], batch[ind]["pitch_std"]
        ]) for ind in cut_list
    ]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitches = pad_1D_tensor(pitches)
    energies = pad_1D_tensor(energies)

    pitch_specs = pad_2D_tensor(pitch_specs)
    pitch_stats = pad_1D_tensor(pitch_stats)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "pitch": pitches,
           "energy": energies,

           "pitch_spec": pitch_specs,
           "pitch_stats": pitch_stats
    }

    return out


def collate_fn_tensor(train_config):
    def wrapper(batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // train_config.batch_expand_size

        cut_list = list()
        for i in range(train_config.batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(train_config.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
    return wrapper
