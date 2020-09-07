from tensorboardX import SummaryWriter
from utils.common_utils import *


class DataLogger(SummaryWriter):
    def __init__(self, logdir):
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            os.chmod(logdir, 0o775)
        super().__init__(logdir)

    def log_training(self, loss_value, loss_name, iteration):
        for _value, _name in zip(loss_value, loss_name):
            if isinstance(_value, list):
                self.add_scalars(_name[0], {s: v for s, v in zip(_name[1:], _value)}, iteration)
            else:
                self.add_scalar(_name, _value, iteration)

    def log_validation(self, spectrogram_src,
                       spectrogram_gen,
                       emb_fig,
                       audio,
                       iteration):
        if audio is not None:
            self.add_audio("audio", audio, iteration, sample_rate=24000)
        self.add_image(
            "spectrogram/synthesize",
            plot_spectrogram_to_numpy(spectrogram_gen),
            iteration,
            dataformats="HWC"
        )
        self.add_image(
            "spectrogram/source",
            plot_spectrogram_to_numpy(spectrogram_src),
            iteration,
            dataformats="HWC"
        )
        if emb_fig is not None:
            self.add_figure("speaker embedding", emb_fig, iteration)
