from parallel_wavegan.models import ParallelWaveGANGenerator
from utils.logger import DataLogger
from sklearn.preprocessing import StandardScaler
from data_utils.mel_dataset import VqVaeDataset, VqVaeCollateFn
import torch
from utils.common_utils import *
from sklearn.decomposition import PCA
from models.vqvae_2stage import VectorQuantisedVAE


class Validator(object):
    def __init__(self, logger: DataLogger, n_speaker=100, target_id=0,
                 test_mel_path="/home/messier/PycharmProjects/data/jvs_ver1/mel24k/jvs001/BASIC5000_0235.npy"):
        self.n_speaker = n_speaker
        self.mel_nat = np.load(test_mel_path)
        self.seq_length = self.mel_nat.shape[-1]
        if (self.mel_nat.shape[-1] % 8) != 0:
            pad_size = 8 - self.mel_nat.shape[-1] % 8
            self.mel_nat = np.pad(self.mel_nat, ((0, 0), (0, pad_size)), mode="edge")

        self.mcc_nat = VqVaeDataset.mel2mcc(self.mel_nat)
        self.mcc_nat = torch.from_numpy(self.mcc_nat).float()

        language_id_en = torch.zeros([1, 2], dtype=torch.float32).cuda()
        language_id_en[:, 0] = 1.
        language_id_jp = torch.zeros([1, 2], dtype=torch.float32).cuda()
        language_id_jp[:, 1] = 1.
        self.language_id = language_id_en

        self.c1 = torch.zeros([1, n_speaker])
        self.c1[0, target_id] = 1.0

        self.logger = logger
        vctk_speaker_info = np.load('/home/messier/PycharmProjects/VAE-VC/data/speaker_info.npy')[:n_speaker]
        vctk_gender = vctk_speaker_info[:, 1]
        self.idx_female = np.array(np.where(vctk_gender == 0))
        self.idx_male = np.array(np.where(vctk_gender == 1))

        wavegan_state_dict = torch.load("/home/messier/PycharmProjects/VCC2020/wavegan_checkpoint_vctk.pkl")

        self.wavegan = ParallelWaveGANGenerator()
        self.wavegan.load_state_dict(wavegan_state_dict["model"]["generator"])
        self.wavegan.remove_weight_norm()
        self.wavegan.eval()
        self.wavegan_scaler = StandardScaler()
        self.wavegan_scaler.mean_ = read_hdf5("/home/messier/PycharmProjects/VCC2020/stats_vctk.h5", "mean")
        self.wavegan_scaler.scale_ = read_hdf5("/home/messier/PycharmProjects/VCC2020/stats_vctk.h5", "scale")
        self.wavegan_scaler.n_features_in_ = self.wavegan_scaler.mean_.shape[0]

    def __call__(self, model: VectorQuantisedVAE, iteration, annotation=None):
        model.eval()
        with torch.no_grad():
            mcc_infer = model.inference([self.mcc_nat.unsqueeze(0).cuda(),
                                         self.c1.cuda(),
                                         self.language_id.cuda()]).detach().squeeze().cpu().numpy()
        mcc_infer = mcc_infer[:, :self.seq_length]

        mel_infer = VqVaeDataset.mcc2mel(mcc_infer)
        mel_infer = mel_infer[:, :self.seq_length]
        mel_infer_norm = self.wavegan_scaler.transform(mel_infer.T).T
        mel_infer_norm = torch.from_numpy(mel_infer_norm).float().unsqueeze(0)
        with torch.no_grad():
            audio = self.wavegan.inference(mel_infer_norm, device=torch.device("cpu")).squeeze().numpy()
        audio = audio / max(abs(audio))

        # emb = model.get_speaker_emb().cpu().numpy()
        # pca_emb = PCA(n_components=2)
        # emb_pca = pca_emb.fit_transform(emb)
        # emb_fig = plt.figure(dpi=200)
        # plt.scatter(emb_pca[:, 0], emb_pca[:, 1], alpha=0.8)
        # plt.grid()
        # if annotation is not None:
        #     for k, v in annotation.items():
        #         plt.scatter(emb_pca[v, 0], emb_pca[v, 1], alpha=0.8, c='red')
        #         plt.annotate(k, (emb_pca[v, 0], emb_pca[v, 1]))

        self.logger.log_validation(spectrogram_gen=mel_infer,
                                   spectrogram_src=self.mel_nat,
                                   emb_fig=None,
                                   audio=audio,
                                   iteration=iteration)
        plt.close()
        model.train()
