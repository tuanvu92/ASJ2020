import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ConvNorm(nn.Conv1d):
    """ 1D convolution layer with padding mode """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 pad_mode="same", dilation=1, groups=1, bias=True, w_init_gain='linear'):
        self.pad_mode = pad_mode
        if pad_mode is "same":
            # assert(kernel_size % 2 == 1)
            _pad = int((dilation * (kernel_size - 1) + 1 - stride) / 2)
        elif pad_mode == "causal":
            _pad = dilation * (kernel_size - 1)
        else:
            _pad = 0
        self._pad = _pad
        super(ConvNorm, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=_pad,
                                       dilation=dilation,
                                       bias=bias,
                                       groups=groups)
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = super(ConvNorm, self).forward(signal)
        if self.pad_mode is "causal":
            conv_signal = conv_signal[:, :, :-self._pad]
        return conv_signal


class GateActivationUnit(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dilation=1):
        super(GateActivationUnit, self).__init__()
        self.in_layer = ConvNorm(input_dim, 2 * hidden_dim,
                                 kernel_size=kernel_size,
                                 dilation=dilation)
        self.out_layer = nn.Sequential(
            ConvNorm(hidden_dim, input_dim, kernel_size=3),
            nn.InstanceNorm1d(input_dim, momentum=0.8)
        )

    def forward(self, x):
        h = self.in_layer(x)
        h_t = torch.tanh(h[:, :h.shape[1]//2, :])
        h_s = torch.sigmoid(h[:, h.shape[1]//2:, :])
        h = h_t * h_s
        skip = self.out_layer(h)
        output = x + skip
        return output, skip


class CondGateActivationUnit(nn.Module):
    def __init__(self, feature_dim, cond_dim, hidden_dim=128, kernel_size=5, dilation=1):
        super(CondGateActivationUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_layer = nn.Sequential(
            ConvNorm(feature_dim,
                     2*hidden_dim,
                     kernel_size=kernel_size,
                     dilation=dilation),
            nn.InstanceNorm1d(2*hidden_dim, momentum=0.25),
        )

        self.cond_layer = nn.Sequential(
            ConvNorm(cond_dim, 2*hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.conv_fuse = ConvNorm(2*hidden_dim, 2*hidden_dim, kernel_size=1, groups=2)

        self.out_layer = nn.Sequential(
            ConvNorm(hidden_dim, feature_dim, kernel_size=3),
            nn.InstanceNorm1d(feature_dim, momentum=0.25)
        )

    def forward(self, inputs):
        x, cond = inputs
        acts = self.conv_fuse(self.in_layer(x) + self.cond_layer(cond))
        tanh_act = torch.tanh(acts[:, :self.hidden_dim, :])
        sigmoid_act = torch.sigmoid(acts[:, self.hidden_dim:, :])
        acts = tanh_act * sigmoid_act
        skip = self.out_layer(acts)
        output = x + skip
        return output, skip


class VQ_layer(nn.Module):
    def __init__(self, emb_dim, n_emb, n_codebook):
        super(VQ_layer, self).__init__()
        self.n_codebook = n_codebook
        self.codebook = Parameter(torch.Tensor(n_codebook, emb_dim, n_emb).uniform_(-1/n_emb, 1/n_emb))
        self.emb_dim = emb_dim

    def forward(self, z_e_x, codebook_index, training=True):
        codebook = torch.index_select(self.codebook, dim=0, index=codebook_index).squeeze(0)
        inputs_size = z_e_x.size()
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous().view(-1, self.emb_dim)
        dist2 = torch.sum(z_e_x_**2, 1, keepdim=True) - 2*torch.matmul(z_e_x_, codebook) \
                + torch.sum(codebook**2, 0)
        _, z_id_flatten = torch.max(-dist2, dim=1)
        z_id = z_id_flatten.view(inputs_size[0], inputs_size[2])
        z_q_flatten = torch.index_select(codebook.t(), dim=0, index=z_id_flatten)
        z_q = z_q_flatten.view(inputs_size[0], inputs_size[2], self.emb_dim).permute(0, 2, 1)

        return z_q, z_id


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_gru=1,
                 down_sample_factor=2, n_layers=5):
        super().__init__()
        self.n_layers = n_layers
        self.input_layer = nn.Sequential(
            ConvNorm(input_dim, 2*hidden_dim, kernel_size=15),
            nn.GLU(dim=1)
        )

        self.down_sample = nn.ModuleList()
        assert down_sample_factor % 2 == 0
        for i in range(down_sample_factor//2):
            self.down_sample.extend([ConvNorm(hidden_dim, 2*hidden_dim, kernel_size=8, stride=2),
                                     nn.InstanceNorm1d(2*hidden_dim, momentum=0.8),
                                     nn.GLU(dim=1)
                                     ])
        self.down_sample = nn.Sequential(*self.down_sample)

        self.dilated_conv = nn.ModuleList()
        for i in range(n_layers):
            self.dilated_conv.append(GateActivationUnit(hidden_dim, hidden_dim, dilation=2**i))
        self.output_layer = nn.Sequential(
            ConvNorm(hidden_dim, 2*output_dim, kernel_size=5),
            nn.InstanceNorm1d(2*output_dim, momentum=0.8),
            nn.GLU(dim=1),
            ConvNorm(output_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.down_sample(h)
        output = None
        for i in range(self.n_layers):
            h, _out = self.dilated_conv[i](h)
            if i == 0:
                output = _out
            else:
                output += _out
        output = self.output_layer(output)
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cond_dim,
                 n_stage, n_gru=1,
                 n_upsample_factor=2):
        super(Decoder, self).__init__()
        self.input_layer = nn.Sequential(
            ConvNorm(input_dim, 2*hidden_dim, kernel_size=5),
            nn.InstanceNorm1d(2*hidden_dim, momentum=0.25),
            nn.GLU(dim=1)
        )

        self.cond_layer = nn.Sequential(
            ConvNorm(cond_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.conv_fuse = ConvNorm(hidden_dim, hidden_dim, kernel_size=3)
        assert n_upsample_factor % 2 == 0
        self.upsample = nn.ModuleList()
        for i in range(n_upsample_factor // 2):
            self.upsample.append(nn.Sequential(nn.ConvTranspose1d(hidden_dim + cond_dim,
                                                                  2*hidden_dim,
                                                                  kernel_size=8,
                                                                  stride=2,
                                                                  padding=3),
                                               nn.InstanceNorm1d(2*hidden_dim, momentum=0.25),
                                               nn.GLU(dim=1)))
        self.dilated_conv = nn.ModuleList()
        dilation_rate = [1, 2, 4, 8, 16, 32]
        for i in range(n_stage):
            for d in dilation_rate:
                self.dilated_conv.append(CondGateActivationUnit(hidden_dim,
                                                                cond_dim,
                                                                hidden_dim,
                                                                kernel_size=5,
                                                                dilation=d))
        self.output_layer = nn.Sequential(
            ConvNorm(hidden_dim, 2*output_dim, kernel_size=5),
            nn.InstanceNorm1d(2*output_dim, momentum=0.25),
            nn.GLU(dim=1),
            ConvNorm(output_dim, output_dim, kernel_size=15),
        )

    def forward(self, inputs):
        x, cond = inputs
        cond_rep = cond.repeat(1, 1, x.size(-1))
        h = self.conv_fuse(self.input_layer(x) + self.cond_layer(cond_rep))
        for upsample in self.upsample:
            h = upsample(torch.cat([h, cond.repeat(1, 1, h.size(-1))], dim=1))

        cond_rep = cond.repeat(1, 1, h.size(-1))
        output = None
        for i in range(len(self.dilated_conv)):
            h, _out = self.dilated_conv[i]([h, cond_rep])
            if i == 0:
                output = _out
            else:
                output += _out
        output = self.output_layer(output)
        return output
