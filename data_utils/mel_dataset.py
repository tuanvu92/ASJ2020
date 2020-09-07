import torch
from torch.utils.data import DataLoader, Dataset
import random
from utils.common_utils import *
from sklearn.preprocessing import StandardScaler
import librosa


class VqVaeDataset(Dataset):
    def __init__(self, mel_dir, n_speaker=100,
                 norm=True, train_file_list=None,
                 use_delta=False,
                 append_vcc2020=True,
                 id_offset=0):
        self.norm = norm
        self.use_delta = use_delta
        self.id_offset = id_offset
        random.seed(12345)
        if train_file_list is None:
            file_list = getListOfFiles(mel_dir)
        else:
            with open("train_file_list.txt", "r") as f:
                file_list = f.read().split('\n')[:-1]
        self.mel_file_list = [fname for fname in file_list if fname.find(".npy") != -1]
        self.speaker_label = self.create_speaker_label(n_speaker)
        self.mel_file_list = [fname for fname in self.mel_file_list
                              if fname.split("/")[-2] in self.speaker_label]
        random.shuffle(self.mel_file_list)
        self.n_speaker = n_speaker
        # For VCC2020 data
        if append_vcc2020:
            self.append_vcc2020_data()
        self.scaler = StandardScaler()
        stats = np.load("mcc_stats.npy")
        self.scaler.mean_ = stats[0]
        self.scaler.scale_ = stats[1]

        # ID offset for cross-lingual training
        for k, v in self.speaker_label.items():
            self.speaker_label[k] = v + self.id_offset

    def __len__(self):
        return len(self.mel_file_list)

    def append_vcc2020_data(self):
        english_speaker = ["SEF1", "SEF2", "SEM1", "SEM2", "TEF1", "TEF2", "TEM1", "TEM2"]
        vcc_mel_dir = "/home/messier/PycharmProjects/data/vcc2020_training/mel24k/"
        vcc_file_list = getListOfFiles(vcc_mel_dir)
        vcc_file_list = [fname for fname in vcc_file_list if fname.split("/")[-2] in english_speaker]
        self.mel_file_list.extend(vcc_file_list)
        random.shuffle(self.mel_file_list)
        for i, speaker_name in enumerate(english_speaker):
            self.speaker_label[speaker_name] = i + self.n_speaker

        self.n_speaker = len(self.speaker_label)

    def create_speaker_label(self, n_speaker):
        speaker_list = []
        for fname in self.mel_file_list:
            # speaker_name = fname.split("/")[-1].split("_")[0]
            speaker_name = fname.split("/")[-2]
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
        speaker_list.sort()
        speaker_list = speaker_list[:n_speaker]
        speaker_label = {spkr_name: int(i) for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_label

    def get_label(self, fname):
        speaker_name = fname.split("/")[-2]
        return self.speaker_label[speaker_name]

    @staticmethod
    def mel2mcc(mel):
        if len(mel.shape) == 2:
            mel = np.expand_dims(mel, 0)
        c = np.fft.irfft(mel, axis=1)
        c[:, 0] /= 2.0
        c = c[:, :mel.shape[1]]
        return np.squeeze(c)

    @staticmethod
    def mcc2mel(mcc):
        if len(mcc.shape) == 2:
            mcc = np.expand_dims(mcc, 0)
        sym_c = np.zeros([mcc.shape[0], 2*(mcc.shape[1]-1), mcc.shape[2]])
        sym_c[:, 0] = 2*mcc[:, 0]
        for i in range(1, mcc.shape[1]):
            sym_c[:, i] = mcc[:, i]
            sym_c[:, -i] = mcc[:, i]
        mel = np.fft.rfft(sym_c, axis=1).real
        mel = np.squeeze(mel)
        return mel

    def __getitem__(self, index):
        mel = np.load(self.mel_file_list[index])
        mcc = self.mel2mcc(mel)
        if self.use_delta:
            mcc_delta = librosa.feature.delta(mcc, axis=0, order=1)
            mcc_delta2 = librosa.feature.delta(mcc, axis=0, order=2)
            mcc = np.concatenate([mcc, mcc_delta, mcc_delta2], axis=0)
        if self.norm:
            mcc = self.scaler.transform(mcc.T).T
        mcc = torch.from_numpy(mcc).float()
        y = torch.zeros([1, self.n_speaker])
        speaker_label = self.get_label(self.mel_file_list[index])
        y[0, speaker_label] = 1.0
        return mcc, y


class VqVaeCollateFn(object):
    def __init__(self, max_seq_len=512):
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # batch = [seq, label]
        # seq.shape = [T,]
        # label.shape = [n_speaker]
        batch_size = len(batch)
        feature_dim = batch[0][0].shape[0]
        label = torch.cat([x[1] for x in batch], dim=0)
        seq_lengths = [x[0].shape[1] for x in batch]
        batch_max_seq_len = min(self.max_seq_len, max(seq_lengths))
        batch_max_seq_len = batch_max_seq_len - batch_max_seq_len % 8
        padded_seq = torch.zeros([batch_size, feature_dim,  batch_max_seq_len])

        for i, (seq, _) in enumerate(batch):
            if batch_max_seq_len < seq.shape[1]:
                # Select random segment from sequence
                start = torch.randint(low=0, high=seq.shape[1] - batch_max_seq_len, size=[1])
                padded_seq[i] = seq[:, start: start + batch_max_seq_len]
            else:
                # Wrap-padding sequence to batch_max_seq_len
                n = batch_max_seq_len // seq.shape[1]
                for j in range(n):
                    padded_seq[i, :, seq.shape[1]*j: seq.shape[1]*(j+1)] = seq
                wrap_len = batch_max_seq_len % seq.shape[1]
                if wrap_len > 0:
                    padded_seq[i, :, -wrap_len:] = seq[:, :wrap_len]

        return padded_seq, label


class CrossLingualDataset(Dataset):
    def __init__(self, en_train_list, jp_train_list, append_vcc2020=True):
        random.seed(12345)
        # Load english dataset
        file_list_en = read_file_list(en_train_list)
        file_list_jp = read_file_list(jp_train_list)
        self.mel_file_list_en = [fname for fname in file_list_en if fname.find(".npy") != -1]
        self.speaker_label_en = self.create_speaker_label(self.mel_file_list_en)
        self.mel_file_list_en = [fname for fname in self.mel_file_list_en
                                 if fname.split("/")[-2] in self.speaker_label_en]
        random.shuffle(self.mel_file_list_en)
        print("n_en_utt: ", len(self.mel_file_list_en))
        print("n_speaker_en: ", len(self.speaker_label_en))

        # Load japanese dataset
        self.mel_file_list_jp = [fname for fname in file_list_jp if fname.find(".npy") != -1]
        self.speaker_label_jp = self.create_speaker_label(self.mel_file_list_jp)
        self.mel_file_list_jp = [fname for fname in self.mel_file_list_jp
                                 if fname.split("/")[-2] in self.speaker_label_jp]
        random.shuffle(self.mel_file_list_jp)
        print("n_en_utt: ", len(self.mel_file_list_jp))
        print("n_speaker_jp:", len(self.speaker_label_jp))

        self.start_idx = np.random.randint(0, max(len(self.speaker_label_en),
                                                  len(self.speaker_label_jp)))

        self.n_speaker = len(self.speaker_label_en) + len(self.speaker_label_jp)

    def __len__(self):
        return min(len(self.mel_file_list_en), len(self.mel_file_list_jp))

    def random_start_idx(self):
        self.start_idx += min(len(self.mel_file_list_en), len(self.mel_file_list_jp))
        self.start_idx = self.start_idx % max(len(self.mel_file_list_en), len(self.mel_file_list_jp))

    @staticmethod
    def create_speaker_label(mel_file_list):
        speaker_list = []
        for fname in mel_file_list:
            speaker_name = fname.split("/")[-2]
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
        speaker_list.sort()
        speaker_label = {spkr_name: int(i) for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_label

    def get_label(self, fname):
        speaker_name = fname.split("/")[-2]
        if speaker_name in self.speaker_label_en:
            return self.speaker_label_en[speaker_name]
        elif speaker_name in self.speaker_label_jp:
            return self.speaker_label_jp[speaker_name] + len(self.speaker_label_en)
        else:
            print("Speaker %s not in either speaker list" % fname)
            return None

    @staticmethod
    def mel2mcc(mel):
        if len(mel.shape) == 2:
            mel = np.expand_dims(mel, 0)
        c = np.fft.irfft(mel, axis=1)
        c[:, 0] /= 2.0
        c = c[:, :mel.shape[1]]
        return np.squeeze(c)

    @staticmethod
    def mcc2mel(mcc):
        if len(mcc.shape) == 2:
            mcc = np.expand_dims(mcc, 0)
        sym_c = np.zeros([mcc.shape[0], 2*(mcc.shape[1]-1), mcc.shape[2]])
        sym_c[:, 0] = 2*mcc[:, 0]
        for i in range(1, mcc.shape[1]):
            sym_c[:, i] = mcc[:, i]
            sym_c[:, -i] = mcc[:, i]
        mel = np.fft.rfft(sym_c, axis=1).real
        mel = np.squeeze(mel)
        return mel

    def get_item_index(self, index, mel_file_list):
        mel = np.load(mel_file_list[index])
        mcc = self.mel2mcc(mel)
        mcc = torch.from_numpy(mcc).float()
        y = torch.zeros([1, self.n_speaker])
        speaker_label = self.get_label(mel_file_list[index])
        y[0, speaker_label] = 1.0
        return mcc, y

    def __getitem__(self, index):
        index_en = (index + self.start_idx) % len(self.mel_file_list_en)
        index_jp = (index + self.start_idx) % len(self.mel_file_list_jp)
        mcc_en, y_en = self.get_item_index(index_en, self.mel_file_list_en)
        mcc_jp, y_jp = self.get_item_index(index_jp, self.mel_file_list_jp)
        return mcc_en, y_en, mcc_jp, y_jp


class CrossLingualCollateFn(object):
    def __init__(self, max_seq_len=512):
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # batch = [seq, label]
        # seq.shape = [T,]
        # label.shape = [n_speaker]
        batch_size = len(batch)
        feature_dim = batch[0][0].shape[0]
        label = torch.cat([x[1] for x in batch] + [x[3] for x in batch], dim=0)

        seq_lengths = [x[0].shape[1] for x in batch] + [x[2].shape[1] for x in batch]

        batch_max_seq_len = min(self.max_seq_len, max(seq_lengths))
        batch_max_seq_len = batch_max_seq_len - batch_max_seq_len % 8
        padded_seq = torch.zeros([2*batch_size, feature_dim,  batch_max_seq_len])

        for i, (seq_en, _, seq_jp, _) in enumerate(batch):
            padded_seq[i] = self.pad(seq_en, feature_dim, batch_max_seq_len)
            padded_seq[i+batch_size] = self.pad(seq_jp, feature_dim, batch_max_seq_len)
        return padded_seq, label

    @staticmethod
    def pad(seq, feature_dim, batch_max_seq_len):
        if batch_max_seq_len < seq.shape[1]:
            # Select random segment from sequence
            start = torch.randint(low=0, high=seq.shape[1] - batch_max_seq_len, size=[1])
            padded_seq = seq[:, start: start + batch_max_seq_len]
        else:
            padded_seq = torch.zeros([feature_dim, batch_max_seq_len])
            # Wrap-padding sequence to batch_max_seq_len
            n = batch_max_seq_len // seq.shape[1]
            for j in range(n):
                padded_seq[:, seq.shape[1] * j: seq.shape[1] * (j + 1)] = seq
            wrap_len = batch_max_seq_len % seq.shape[1]
            if wrap_len > 0:
                padded_seq[:, -wrap_len:] = seq[:, :wrap_len]
        return padded_seq


class CollateNoTruncate(object):
    def __init__(self):
        return

    def __call__(self, batch):
        batch_size = len(batch)
        feature_dim = batch[0][0].shape[0]
        label = torch.cat([x[1] for x in batch], dim=0)
        seq_lengths = [x[0].shape[1] for x in batch]
        batch_max_seq_len = max(seq_lengths)
        padded_seq = torch.zeros([batch_size, feature_dim, batch_max_seq_len])
        for i, (seq, _) in enumerate(batch):
            if batch_max_seq_len < seq.shape[1]:
                # Select random segment from sequence
                start = torch.randint(low=0, high=seq.shape[1] - batch_max_seq_len, size=[1])
                padded_seq[i] = seq[:, start: start + batch_max_seq_len]
            else:
                # Wrap-padding sequence to batch_max_seq_len
                n = batch_max_seq_len // seq.shape[1]
                for j in range(n):
                    padded_seq[i, :, seq.shape[1] * j: seq.shape[1] * (j + 1)] = seq
                wrap_len = batch_max_seq_len % seq.shape[1]
                if wrap_len > 0:
                    padded_seq[i, :, -wrap_len:] = seq[:, :wrap_len]
        return padded_seq, label, seq_lengths
