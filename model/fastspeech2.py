import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from variance_adaptor import VarianceAdaptor

from utils import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, model_config, melspec_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config, melspec_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, melspec_config.num_mels)

    def mask_tensor(self, tnsr, position, mel_max_length):
        """Set predicted data values to zero, where real data was padded"""
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, tnsr.size(-1))
        return tnsr.masked_fill(mask, 0.)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        pitch_target=None,
        # pitch_spec_target=None,
        # pitch_stats_target=None,
        energy_target=None,
        length_coef=1.0,
        pitch_coef=1.0,
        energy_coef=1.0
    ):
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            (
                lr_output,
                duration_predictor_output,
                pitch_predictor_output,
                # pitch_spec_predictor_output,
                # pitch_stats_predictor_output,
                energy_predictor_output
             ) = self.variance_adaptor(
                 enc_output,
                 length_coef,
                 pitch_coef,
                 energy_coef,
                 length_target,
                 pitch_target,
                #  pitch_spec_target,
                #  pitch_stats_target,
                 energy_target,
                 mel_max_length
            )
            dec_output = self.decoder(lr_output, mel_pos)
            output = self.mel_linear(dec_output)

            output = self.mask_tensor(output, mel_pos, mel_max_length)

            pitch_predictor_output = self.mask_tensor(
                pitch_predictor_output.unsqueeze(-1),
                mel_pos, mel_max_length
            ).squeeze() # зануляем западденные питчи

            # pitch_spec_predictor_output = self.mask_tensor(
            #     pitch_spec_predictor_output,
            #     mel_pos, mel_max_length
            # ).squeeze()  # зануляем западденные питчи

            energy_predictor_output = self.mask_tensor(
                energy_predictor_output.unsqueeze(-1),
                mel_pos, mel_max_length
            ).squeeze() # зануляем западденные энергии

            return output, duration_predictor_output, pitch_predictor_output, energy_predictor_output
            # pitch_spec_predictor_output, pitch_stats_predictor_output, energy_predictor_output

        else:
            (
                lr_output,
                duration_predictor_output
             ) = self.variance_adaptor(
                enc_output,
                length_coef,
                pitch_coef,
                energy_coef
            )
            dec_output = self.decoder(lr_output, duration_predictor_output)
            output = self.mel_linear(dec_output)
            return output
