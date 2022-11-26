import torch.nn as nn

from utils import Transpose
from configs import FastSpeechConfig

class FeaturePredictor(nn.Module):
    """ Duration/Energy Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super().__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class PitchSpectrogramPredictor(nn.Module):
    def __init__(self, model_config: FastSpeechConfig):
        super().__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 10)


    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class PitchMeanStdPredictor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(model_config.encoder_dim, model_config.encoder_dim),
            nn.ReLU(),
            nn.Linear(model_config.encoder_dim, model_config.encoder_dim),
            nn.ReLU(),
            nn.Linear(model_config.encoder_dim, 2),
            nn.ReLU()
        )

    def forward(self, hidden_state):
        x = hidden_state.sum(dim=-2)
        return self.projector(x)

