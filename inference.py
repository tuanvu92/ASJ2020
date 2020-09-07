import numpy as np
import argparse
from data_utils.mel_dataset import *
from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator
from scipy.io.wavfile import write as wave_write
from os.path import join, exists
import json
from progressbar import progressbar

from models.vqvae_3stage import VectorQuantisedVAE as vqvae_3stage
from models.vqvae_2stage import VectorQuantisedVAE as vqvae_2stage
from models.vqvae_1stage import VectorQuantisedVAE as vqvae_1stage


def inference(mode, checkpoint_path, lang,
              speaker_label_json,
              source_speaker, target_speaker,
              test_file_list, dst_path):
    if not exists(dst_path):
        os.makedirs(dst_path)
    with open(speaker_label_json, 'r') as f:
        speaker_label = json.load(f)
    target_speaker_list = list(target_speaker.keys())
    if mode == 1:
        model = vqvae_1stage(codebook_size_bot=384, n_speaker=len(speaker_label), n_codebook=1)
    elif mode == 2:
        model = vqvae_2stage(codebook_size_top=192, codebook_size_bot=192,
                             downsampling_top=4, downsampling_bot=2,
                             n_speaker=len(speaker_label), n_codebook=1)
    elif mode == 3:
        model = vqvae_3stage(codebook_size_top=128, codebook_size_mid=128,
                             codebook_size_bot=128,
                             n_speaker=len(speaker_label), n_codebook=1)
    elif mode == 4:
        model = vqvae_3stage(codebook_size_top=128, codebook_size_mid=128,
                             codebook_size_bot=128,
                             n_speaker=len(speaker_label), n_codebook=2)

    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda().eval()
    print(target_speaker)
    c = torch.zeros((len(target_speaker), len(speaker_label))).cuda()
    for i, tar in enumerate(target_speaker_list):
        c[i, target_speaker[tar]] = 1.0

    language_id = torch.zeros((len(target_speaker), 2), dtype=torch.float32).cuda()
    language_id[:, lang] = 1.

    print("Loading wavegan model...")
    wavegan_state_dict = torch.load("/home/messier/PycharmProjects/VCC2020/wavegan_checkpoint_vctk.pkl")

    wavegan = ParallelWaveGANGenerator()
    wavegan.load_state_dict(wavegan_state_dict["model"]["generator"])
    wavegan.remove_weight_norm()
    wavegan.cuda().eval()
    wavegan_scaler = StandardScaler()
    wavegan_scaler.mean_ = read_hdf5("/home/messier/PycharmProjects/VCC2020/stats_vctk.h5", "mean")
    wavegan_scaler.scale_ = read_hdf5("/home/messier/PycharmProjects/VCC2020/stats_vctk.h5", "scale")
    wavegan_scaler.n_features_in_ = wavegan_scaler.mean_.shape[0]

    for src_spkr in source_speaker.keys():
        print(src_spkr)
        mel_file_list = read_file_list(test_file_list)
        mel_file_list = [f for f in mel_file_list if f.find(".npy") != -1 and f.find(src_spkr) != -1]
        print("No test file: ", len(mel_file_list))

        for mel_file in progressbar(mel_file_list):
            audio_fname = mel_file.replace('.npy', '').split('/')[-1]
            mel_src = np.load(mel_file)
            seq_length = mel_src.shape[-1]
            if (mel_src.shape[-1] % 8) != 0:
                pad_size = 8 - mel_src.shape[-1] % 8
                mel_src = np.pad(mel_src, ((0, 0), (0, pad_size)), mode="wrap")

            mcc_src = VqVaeDataset.mel2mcc(mel_src)
            mcc_src = torch.from_numpy(mcc_src).float().cuda().unsqueeze(0)
            mcc_src = mcc_src.repeat([c.shape[0], 1, 1])
            with torch.no_grad():
                mcc_infer = model.inference([mcc_src, c, language_id]).cpu().numpy()
            mcc_infer[:, 3:50] *= 1.2

            mel_infer = VqVaeDataset.mcc2mel(mcc_infer)
            mel_infer = mel_infer[:, :, :seq_length]
            mel_infer = wavegan_scaler.transform(
                mel_infer.transpose([0, 2, 1]).reshape([-1, 80])
            ).reshape([mcc_infer.shape[0], -1, 80]).transpose([0, 2, 1])

            mel_infer = torch.from_numpy(mel_infer).float().cuda()

            with torch.no_grad():
                audio = wavegan.inference(mel_infer).squeeze().cpu().numpy()
            for au, tar_spkr in zip(audio, target_speaker_list):
                au = 0.5 * 32768 * au / max(abs(au))
                au = au.astype(np.int16)
                output_dir = join(dst_path, "%s_%s" % (src_spkr, tar_spkr))
                if not exists(output_dir):
                    os.makedirs(output_dir)
                wave_write(join(output_dir, "%s_%s_%s.wav" % (tar_spkr, src_spkr, audio_fname)),
                           24000, au)
    print("Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, default=1,
                        help='select model. 1: vqvae_1stage, '
                             '2: vqvae_2stage, 3: vqvae_3stage')
    parser.add_argument('-c', '--checkpoint', type=str, default='',
                        help='path to checkpoint')
    parser.add_argument('-s', '--speaker_label', type=str, default='', help='speaker label file')
    parser.add_argument('-d', '--dst_dir', type=str, default='', help='output dir')
    args = parser.parse_args()

    inference(args.mode, args.checkpoint, 0, args.speaker_label,
              {"p225": 0, "p226": 1},
              {"jvs001": 100, "jvs002": 101},
              "file_lists/vctk_eval_list.txt",
              dst_path="gen/%d/" % args.mode)

    inference(args.mode, args.checkpoint, 1, args.speaker_label,
              {"jvs001": 100, "jvs002": 101},
              {"p225": 0, "p226": 1},
              "file_lists/jvs_eval_list.txt",
              dst_path="gen/%d/" % args.mode)
