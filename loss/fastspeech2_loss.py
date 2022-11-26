import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel,
        duration_predicted,
        pitch_predicted,
        # pitch_spec_predicted,
        # pitch_stats_predicted,
        energy_predicted,
        mel_target,
        duration_predictor_target,
        pitch_target,
        # pitch_spec_target,
        # pitch_stats_target,
        energy_target
    ):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(
                                        duration_predicted,
                                        torch.log(duration_predictor_target.float() + 1)
                                  )

        pitch_loss = self.mse_loss(
            pitch_predicted,
            torch.log(pitch_target + 1)
        )
        # pitch_spec_loss = self.mse_loss(
        #     pitch_spec_predicted,
        #     pitch_spec_target
        # )
        # pitch_stats_loss = self.mse_loss(
        #     pitch_stats_predicted,
        #     pitch_stats_target
        # )

        energy_loss = self.mse_loss(
            energy_predicted,
            torch.log(energy_target + 1)
        )

        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss
        # pitch_spec_loss, pitch_stats_loss, energy_loss
