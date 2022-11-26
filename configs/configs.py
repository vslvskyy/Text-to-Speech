import torch

from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    num_mels = 80
    min_pitch = 100.0
    max_pitch = 750.0
    # min_log_pitch = 4.
    # max_log_pitch = 6.
    min_energy = 0.0
    max_energy = 100.0


@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    pitch_ground_truth = "./ljspeech_preprocessed_pitches.npy"
    energy_ground_truth = "./ljspeech_energies.npy"
    alignment_path = "./alignments"
    data_path = "./data/train.txt"
    wavs_path = "./data/LJSpeech-1.1/wavs"
    hop_length = 256
    sample_rate = 22050
    n_fft = 1024

    wandb_project = 'fastspeech2'

    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32
