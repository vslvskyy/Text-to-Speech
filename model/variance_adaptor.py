import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_predictors import FeaturePredictor

from utils import create_alignment
from configs import FastSpeechConfig, TrainConfig, MelSpectrogramConfig


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, model_config: FastSpeechConfig, train_config: TrainConfig):
        super().__init__()
        
        self.duration_predictor = FeaturePredictor(model_config)
        self.device = train_config.device

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            int_duration_predictor_output = (((torch.exp(duration_predictor_output) - 1) * alpha) + 0.5).long()
            output = self.LR(x, int_duration_predictor_output, mel_max_length)

            mel_pos = torch.stack(
                [torch.Tensor([i + 1 for i in range(output.size(1))])]
            ).long().to(self.device)
            return output, mel_pos


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(
        self,
        model_config: FastSpeechConfig,
        train_config: TrainConfig,
        melspec_config: MelSpectrogramConfig
    ):
        super().__init__()

        self.pitch_predictor = FeaturePredictor(model_config)
        # self.pitch_spec_predictor = PitchSpectrogramPredictor(model_config)
        # self.pitch_stats_predictor = PitchMeanStdPredictor(model_config)
        self.energy_predictor = FeaturePredictor(model_config)
        self.length_regulator = LengthRegulator(model_config, train_config)

        self.pitch_embedding = nn.Embedding(256, model_config.encoder_dim)
        self.energy_embedding = nn.Embedding(256, model_config.encoder_dim)

        self.pitch_buckets = torch.linspace(melspec_config.min_pitch, melspec_config.max_pitch, 256-1)
        self.energy_buckets = torch.linspace(melspec_config.min_energy, melspec_config.max_energy, 256-1)

    def forward(
        self,
        encoder_output,
        length_coef=1.0,
        pitch_coef=1.0,
        energy_coef=1.0,
        length_target=None,
        pitch_target=None,
        # pitch_spec_target=None,
        # pitch_stats_target=None,
        energy_target=None,
        mel_max_length=None
    ):
        lr_output, duration_prediction = self.length_regulator(encoder_output, length_coef, length_target, mel_max_length)

        # log_pitch_spec_prediction = self.pitch_spec_predictor(lr_output)
        # log_pitch_stats_prediciton = self.pitch_stats_predictor(lr_output)
        # log_pitch_prediction = (
        #     get_pitch_from_spec(log_pitch_spec_prediction).T * log_pitch_stats_prediciton[:, 1] + log_pitch_stats_prediciton[:, 0]
        # ).T

        log_pitch_prediction = self.pitch_predictor(lr_output)

        log_energy_prediction = self.energy_predictor(lr_output)

        if self.training:
            # log_pitch_target = (
            #     get_pitch_from_spec(pitch_spec_target).T * pitch_stats_target[:, 1] + pitch_stats_target[:, 0]
            # ).T
            # log_pitch_target = torch.log(pitch_target + 1)

            # pitch_target = (torch.exp(log_pitch_target) - 1)
            quantized_pitch = torch.bucketize(pitch_target, self.pitch_buckets.to(pitch_target.device))
            pitch_embeddings = self.pitch_embedding(quantized_pitch)

            quantized_energy = torch.bucketize(energy_target, self.energy_buckets.to(energy_target.device))
            energy_embeddings = self.energy_embedding(quantized_energy)

            result = lr_output + pitch_embeddings + energy_embeddings
            return result, duration_prediction, log_pitch_prediction, log_energy_prediction
            # log_pitch_spec_prediction, log_pitch_stats_prediciton, log_energy_prediction
        else:
            pitch_prediction = (torch.exp(log_pitch_prediction) - 1) * pitch_coef
            quantized_pitch = torch.bucketize(pitch_prediction, self.pitch_buckets.to(pitch_prediction.device))
            pitch_embeddings = self.pitch_embedding(quantized_pitch)

            energy_prediction = (torch.exp(log_energy_prediction) - 1) * energy_coef
            quantized_energy = torch.bucketize(energy_prediction, self.energy_buckets.to(energy_prediction.device))
            energy_embeddings = self.energy_embedding(quantized_energy)

            result = lr_output + pitch_embeddings + energy_embeddings

            return result, duration_prediction
