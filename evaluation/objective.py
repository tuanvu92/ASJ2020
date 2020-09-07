from utils.common_utils import *
from utils.stft import TacotronSTFT
import librosa
from data_utils.mel_dataset import VqVaeDataset
from tqdm import tqdm
import json
from models.vqvae_3stage import VectorQuantisedVAE as vqvae_3stage
from models.vqvae_2stage import VectorQuantisedVAE as vqvae_2stage
from models.vqvae_1stage import VectorQuantisedVAE as vqvae_1stage
import matplotlib
import matplotlib.pyplot as plt
font = {'size': 12}
matplotlib.rc('font', **font)


def calculate_model_mcd(model, speaker_label, mel_file_list, lang):
    model.eval()
    mcd = []
    _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    for mel_file in tqdm(mel_file_list):
        speaker = mel_file.split('/')[-2]
        id = speaker_label[speaker]
        c = torch.zeros((1, len(speaker_label))).cuda()
        c[0, id] = 1.0
        language_id = torch.zeros((1, 2), dtype=torch.float32).cuda()
        language_id[:, lang] = 1.
        mel_src = np.load(mel_file)
        mcc_src = VqVaeDataset.mel2mcc(mel_src)
        mcc_src = torch.from_numpy(mcc_src).float().cuda().unsqueeze(0)
        with torch.no_grad():
            mcc_infer = model.inference([mcc_src, c, language_id], return_zid=False)
            mcc_src = mcc_src[:, :, :mcc_infer.shape[-1]]
            _mcd = torch.mean(torch.sqrt(torch.sum((mcc_src - mcc_infer)**2, dim=1)))
            _mcd = _logdb_const * _mcd.cpu().numpy()
            mcd.append(_mcd)

    mcd_mean = np.mean(mcd)
    mcd_std = np.std(mcd)
    return mcd_mean, mcd_std


def calculate_mcd(checkpoint_1, checkpoint_2, checkpoint_3, speaker_label_json, file_list, lang):
    with open(speaker_label_json, "r") as f:
        speaker_label = json.load(f)

    mel_file_list = read_file_list(file_list)
    model = vqvae_1stage(codebook_size_bot=384, n_speaker=len(speaker_label), n_codebook=1)
    model.load_state_dict(torch.load(checkpoint_1))
    model.cuda()
    mcd_mean_1, mcd_std_1 = calculate_model_mcd(model, speaker_label, mel_file_list, lang)
    print("VQVAE-1stage: ", mcd_mean_1, mcd_std_1)

    model = vqvae_2stage(codebook_size_top=192, codebook_size_bot=192,
                         downsampling_top=4, downsampling_bot=2,
                         n_speaker=len(speaker_label), n_codebook=1)
    model.load_state_dict(torch.load(checkpoint_2))
    model.cuda()
    mcd_mean_2, mcd_std_2 = calculate_model_mcd(model, speaker_label, mel_file_list, lang)
    print("VQVAE-2stage: ", mcd_mean_2, mcd_std_2)

    model = vqvae_3stage(codebook_size_top=128, codebook_size_mid=128,
                         codebook_size_bot=128,
                         n_speaker=len(speaker_label), n_codebook=1)
    model.load_state_dict(torch.load(checkpoint_3))
    model.cuda()
    mcd_mean_3, mcd_std_3 = calculate_model_mcd(model, speaker_label, mel_file_list, lang)
    print("VQVAE-3stage: ", mcd_mean_3, mcd_std_3)
    return [mcd_mean_1, mcd_mean_2, mcd_mean_3], [mcd_std_1, mcd_std_2, mcd_std_3]


def plot_ms(file_path_list, label_list, line_style=None, name=None, title=None):
    plt.figure(figsize=(6, 2.5), dpi=100)
    plt.grid(linestyle=':')
    if line_style is None:
        line_style = ['-'] * len(file_path_list)

    if name is None:
        name = 'ms.png'
    stft = TacotronSTFT(2048, 300, 1200, 80, 24000, 80, 7600).cuda(0)
    ms_all = []
    for i, (file_list, label) in enumerate(zip(file_path_list, label_list)):
        # file_list = getListOfFiles(fp)
        ms = []
        for fname in file_list:
            if fname.find(".npy") != -1:
                mel = np.load(fname)
            else:
                x, fs = librosa.load(fname, sr=None)
                x = x / max(abs(x))
                x = torch.from_numpy(x).float().unsqueeze(0).cuda()
                mel = stft.mel_spectrogram(x).cpu().numpy().squeeze()
            mcc = VqVaeDataset.mel2mcc(mel)
            _ms = np.log(np.abs(np.fft.rfft(mcc, n=4096, axis=1)))[3]
            ms.append(_ms)
        ms = np.array(ms)
        ms = np.mean(ms, axis=0)
        ms_all.append(ms)
        # freq_axis = (fs/300)*np.arange(len(ms)) / len(ms)
        plt.plot(ms, label=label, alpha=0.8, linestyle=line_style[i], linewidth=0.6)
        # print(np.mean(ms[:int(32*len(ms)/100)]))

    # plt.xlim([0, 20])
    # plt.ylim([-1, 2])
    plt.xlabel('Modulation frequency index')
    plt.ylabel('MS [dB]')

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    # plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0))
    plt.legend(ncol=2)
    plt.savefig('../gen/' + name)
    for _ms in ms_all[1:]:
        rmse = np.sqrt(np.mean((_ms - ms_all[0])**2))
        print(np.round(rmse, 3))


def ms_test(source, target):
    eval_file_list = read_file_list("../file_lists/vctk_eval_list.txt")
    eval_file_list.extend(read_file_list("../file_lists/jvs_eval_list.txt"))
    for src in source:
        for tar in target:
            print(src, tar)
            target_file_list = [fp for fp in eval_file_list if fp.find(tar) != -1]
            plot_ms([
                target_file_list,
                getListOfFiles('../gen/1/%s_%s' % (src, tar)),
                getListOfFiles('../gen/2/%s_%s' % (src, tar)),
                getListOfFiles('../gen/3/%s_%s' % (src, tar)),
            ],
                label_list=["Target",
                            "1 stage",
                            "2 stages",
                            "3 stages",
                            ],
                line_style=["solid", "solid", "solid", "solid"],
                name='ms_%s_%s.eps' % (src, tar),
                title="%s-%s" % (src, tar)
            )


def plot_mcd_hbar(mcd_1=[1.216689229494903, 0.1460889598075886],
                  mcd_2=[1.179676505831339, 0.1539162424133541],
                  mcd_3=[1.0740255022480942, 0.1208698776390384]):
    plt.figure(dpi=200, figsize=(6, 2))
    mcd_mean = [mcd_1[0], mcd_2[0], mcd_3[0]]
    mcd_err = [mcd_1[1], mcd_2[1], mcd_3[1]]

    plt.barh([2, 1, 0], mcd_mean, xerr=mcd_err, color="C1", edgecolor="black", zorder=1)
    plt.yticks([2, 1, 0], ["Vanilla VQ-VAE", "2-stage VQ-VAE", "3-stage VQ-VAE"])
    plt.xlim([0.6, 1.4])
    plt.xlabel("MCD [dB]")
    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig("../gen/mcd.eps")
    plt.close()


if __name__ == "__main__":
    ms_test(source=["p225", "p226"],
            target=["jvs002", "jvs001"])
    ms_test(source=["jvs002", "jvs001"],
            target=["p225", "p226"])
    # calculate_mcd("../checkpoints/20200714_0800_vqvae_1_stage_1_cb/vqvae_en_jp_weight_30000.pt",
    #               "../checkpoints/20200714_0805_vqvae_2_stage_1_cb/vqvae_en_jp_weight_30000.pt",
    #               "../checkpoints/20200714_1300_vqvae_3_stage_1_cb/vqvae_en_jp_weight_30000.pt",
    #               "../checkpoints/20200714_1300_vqvae_3_stage_1_cb/speaker_label.json",
    #               "../file_lists/vctk_eval_list.txt",
    #               lang=0)
    # plot_mcd_hbar()


