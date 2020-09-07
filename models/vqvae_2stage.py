import numpy as np
from models.layers import *


class VectorQuantisedVAE(nn.Module):
    def __init__(self, mcc_dim=80, top_emb_dim=32, bot_emb_dim=32, codebook_size_top=128,
                 codebook_size_bot=128, hidden_dim=128, n_codebook=1, n_speaker=100,
                 downsampling_top=2, downsampling_bot=2, speaker_emb_dim=16,
                 n_lang=2, lang_emb_dim=16):
        super().__init__()
        self.n_speaker = n_speaker
        self.speaker_emb_dim = speaker_emb_dim
        self.codebook_size_top = codebook_size_top
        self.codebook_size_bot = codebook_size_bot
        self.mcc_dim = mcc_dim
        self.n_codebook = n_codebook
        self.in_channel_dim = mcc_dim

        self.encoder_top = Encoder(input_dim=hidden_dim,
                                   hidden_dim=hidden_dim,
                                   output_dim=top_emb_dim,
                                   down_sample_factor=downsampling_top)

        self.encoder_bot = Encoder(input_dim=self.in_channel_dim,
                                   hidden_dim=hidden_dim,
                                   output_dim=hidden_dim,
                                   down_sample_factor=downsampling_bot)

        self.decoder_top = Decoder(input_dim=top_emb_dim,
                                   hidden_dim=hidden_dim,
                                   output_dim=hidden_dim,
                                   cond_dim=speaker_emb_dim + lang_emb_dim,
                                   n_stage=1,
                                   n_upsample_factor=downsampling_top)

        self.decoder_bot = Decoder(input_dim=top_emb_dim + bot_emb_dim,
                                   hidden_dim=256,
                                   output_dim=self.mcc_dim,
                                   cond_dim=speaker_emb_dim + lang_emb_dim,
                                   n_stage=3,
                                   n_upsample_factor=downsampling_bot)

        self.upsample_t = nn.ConvTranspose1d(in_channels=top_emb_dim,
                                             out_channels=top_emb_dim,
                                             kernel_size=4*downsampling_top,
                                             stride=downsampling_top,
                                             padding=3*downsampling_top//2)

        self.quantize_conv_bot = ConvNorm(in_channels=2*hidden_dim,
                                          out_channels=bot_emb_dim,
                                          kernel_size=3)

        self.quantize_top = VQ_layer(top_emb_dim, codebook_size_top, n_codebook)
        self.quantize_bot = VQ_layer(bot_emb_dim, codebook_size_bot, n_codebook)

        self.speaker_emb_layer = nn.Linear(n_speaker, speaker_emb_dim, bias=False)
        self.language_emb_layer = nn.Linear(n_lang, lang_emb_dim, bias=False)

    def forward(self, inputs):
        x, speaker_id, language_id = inputs

        speaker_emb = self.speaker_emb_layer(speaker_id).unsqueeze(2)
        language_emb = self.language_emb_layer(language_id).unsqueeze(2)
        s = torch.cat([speaker_emb, language_emb], dim=1)
        if self.n_codebook > 1:
            codebook_id = torch.argmax(language_id, dim=-1)[0]
            codebook_id = codebook_id.cuda()
        else:
            codebook_id = torch.tensor(0, dtype=torch.long).cuda()

        h_bot = self.encoder_bot(x)
        h_top = self.encoder_top(h_bot)

        z_top, z_id_top = self.quantize_top(h_top, codebook_id)
        z_top_st = h_top + (z_top - h_top).detach()
        z_top_dec = self.decoder_top([z_top_st, s])

        h_bot_q = self.quantize_conv_bot(torch.cat([h_bot, z_top_dec], dim=1))
        z_bot, z_id_bot = self.quantize_bot(h_bot_q, codebook_id)
        z_bot_st = h_bot_q + (z_bot - h_bot_q).detach()

        z = torch.cat([z_bot_st, self.upsample_t(z_top_st)], dim=1)
        x_hat = self.decoder_bot([z, s])

        mel_org = self.mcc2mel(x[:, :self.mcc_dim, :])
        mel_hat = self.mcc2mel(x_hat)

        rc_loss = (x[:, :self.mcc_dim, :] - x_hat).pow(2).mean()
        mel_loss = (mel_org - mel_hat).pow(2).mean()
        vq_loss = (h_top - z_top.detach()).pow(2).mean() + (h_bot_q - z_bot.detach()).pow(2).mean()
        commitment_loss = (h_top.detach()-z_top).pow(2).mean() + (h_bot_q.detach()-z_bot).pow(2).mean()

        perplexity_top = self.calculate_perplexity(z_id_top, self.codebook_size_top)
        perplexity_mid = torch.tensor(0).cuda()
        perplexity_bot = self.calculate_perplexity(z_id_bot, self.codebook_size_bot)

        loss = rc_loss + vq_loss + 0.25*commitment_loss

        return loss, [rc_loss, mel_loss, vq_loss, commitment_loss,
                      perplexity_top, perplexity_mid, perplexity_bot]

    def inference(self, inputs, return_zid=False):
        x, speaker_id, language_id = inputs
        speaker_emb = self.speaker_emb_layer(speaker_id).unsqueeze(2)
        language_emb = self.language_emb_layer(language_id).unsqueeze(2)
        s = torch.cat([speaker_emb, language_emb], dim=1)
        if self.n_codebook > 1:
            codebook_id = torch.argmax(language_id, dim=-1)[0]
            codebook_id = codebook_id.cuda()
        else:
            codebook_id = torch.tensor(0, dtype=torch.long).cuda()

        h_bot = self.encoder_bot(x)
        h_top = self.encoder_top(h_bot)

        z_top, z_id_top = self.quantize_top(h_top, codebook_id)

        z_top_dec = self.decoder_top([z_top, s])
        h_bot = h_bot[:, :, :z_top_dec.shape[2]]
        h_bot = self.quantize_conv_bot(torch.cat([h_bot, z_top_dec], dim=1))
        z_bot, z_id_bot = self.quantize_bot(h_bot, codebook_id)

        z = torch.cat([z_bot, self.upsample_t(z_top)], dim=1)
        x_hat = self.decoder_bot([z, s])

        if return_zid:
            return x_hat, z_id_top, z_id_bot
        else:
            return x_hat

    def get_speaker_emb(self):
        return self.speaker_emb_layer.weight.data.T

    def set_speaker_emb(self, speaker_emb):
        self.speaker_emb_layer.weight = nn.Parameter(speaker_emb)

    def get_codebook(self):
        codebook_top = self.quantize_top.codebook.data
        codebook_bot = self.quantize_bot.codebook.data
        return codebook_top, codebook_bot

    def set_codebook(self, codebook_top, codebook_bot):
        assert self.quantize_top.codebook.shape == codebook_top.shape
        assert self.quantize_bot.codebook.shape == codebook_bot.shape
        self.quantize_top.codebook = nn.Parameter(codebook_top)
        self.quantize_bot.codebook = nn.Parameter(codebook_bot)

    @staticmethod
    def calculate_perplexity(z_id, codebook_size):
        z_id = z_id.flatten()
        z_id_onehot = torch.eye(codebook_size, dtype=torch.float32).cuda().index_select(dim=0, index=z_id)
        avg_probs = z_id_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def mcc2mel(self, mcc):
        mcc = mcc.transpose(1, 2)
        mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
        mcc[:, :, 0] = mcc[:, :, 0] * 2.0
        mel = torch.rfft(mcc, signal_ndim=1)
        mel = mel[:, :, :, 0]
        return mel.transpose(1, 2)

    def copy_state_dict(self, checkpoint_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint_path)
        for k, v in pretrained_dict.items():
            if (k in model_dict) and (model_dict[k].shape == v.shape):
                model_dict[k] = v
            else:
                print(k)
        self.load_state_dict(model_dict)