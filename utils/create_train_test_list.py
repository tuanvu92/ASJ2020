from utils.common_utils import getListOfFiles
from tqdm import tqdm


def get_speaker_list(file_list, pos=-2):
    speaker_label = []
    for fname in file_list:
        sp = fname.split("/")[pos]
        if sp not in speaker_label:
            speaker_label.append(sp)
    speaker_label.sort()
    return speaker_label


def create_train_test_list(train_utt_per_speaker=100,
                           test_utt_per_speaker=10,
                           eval_utt_per_speaker=10,
                           n_vctk_speaker=100,
                           n_jvs_speaker=100):
    vctk_train_file_list = []
    vctk_test_file_list = []
    vctk_eval_file_list = []
    jvs_train_file_list = []
    jvs_test_file_list = []
    jvs_eval_file_list = []

    vctk_file_list = getListOfFiles("/home/messier/PycharmProjects/data/VCTK/mel24k/")
    vctk_file_list = [fname for fname in vctk_file_list if fname.find(".npy") != -1]
    vctk_speaker_list = get_speaker_list(vctk_file_list)
    vctk_speaker_list = vctk_speaker_list[:n_vctk_speaker]
    vctk_file_list = [fname for fname in vctk_file_list if fname.split("/")[-2] in vctk_speaker_list]

    vctk_speaker_n_train_utt = {sp: 0 for sp in vctk_speaker_list}
    vctk_speaker_n_test_utt = {sp: 0 for sp in vctk_speaker_list}
    vctk_speaker_n_eval_utt = {sp: 0 for sp in vctk_speaker_list}

    for fname in tqdm(vctk_file_list):
        sp = fname.split("/")[-2]
        if vctk_speaker_n_train_utt[sp] < train_utt_per_speaker:
            vctk_train_file_list.append(fname)
            vctk_speaker_n_train_utt[sp] += 1
        elif vctk_speaker_n_test_utt[sp] < test_utt_per_speaker:
            vctk_test_file_list.append(fname)
            vctk_speaker_n_test_utt[sp] += 1
        elif vctk_speaker_n_eval_utt[sp] < eval_utt_per_speaker:
            vctk_eval_file_list.append(fname)
            vctk_speaker_n_eval_utt[sp] += 1

    print("VCTK train and test list exceeded maximum number")
    print("train list size: ", len(vctk_train_file_list))
    print("test list size: ", len(vctk_test_file_list))

    jvs_file_list = getListOfFiles("/home/messier/PycharmProjects/data/jvs_ver1/mel24k/")
    jvs_file_list = [fname for fname in jvs_file_list if fname.find(".npy") != -1]
    jvs_speaker_list = get_speaker_list(jvs_file_list)[:n_jvs_speaker]
    jvs_file_list = [fname for fname in jvs_file_list if fname.split("/")[-2] in jvs_speaker_list]

    jvs_speaker_n_train_utt = {sp: 0 for sp in jvs_speaker_list}
    jvs_speaker_n_test_utt = {sp: 0 for sp in jvs_speaker_list}
    jvs_speaker_n_eval_utt = {sp: 0 for sp in jvs_speaker_list}
    for fname in tqdm(jvs_file_list):
        sp = fname.split("/")[-2]
        if jvs_speaker_n_train_utt[sp] < train_utt_per_speaker:
            jvs_train_file_list.append(fname)
            jvs_speaker_n_train_utt[sp] += 1
        elif jvs_speaker_n_test_utt[sp] < test_utt_per_speaker:
            jvs_test_file_list.append(fname)
            jvs_speaker_n_test_utt[sp] += 1
        elif jvs_speaker_n_eval_utt[sp] < eval_utt_per_speaker:
            jvs_eval_file_list.append(fname)
            jvs_speaker_n_eval_utt[sp] += 1

    print("JVS train and test list exceeded maximum number")
    print("train list size: ", len(jvs_train_file_list))
    print("test list size: ", len(jvs_test_file_list))

    with open("../file_lists/vctk_train_list.txt", "w") as f:
        for fname in vctk_train_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/vctk_test_list.txt", "w") as f:
        for fname in vctk_test_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/vctk_eval_list.txt", "w") as f:
        for fname in vctk_eval_file_list:
            f.write("%s\n" % fname)

    with open("../file_lists/jvs_train_list.txt", "w") as f:
        for fname in jvs_train_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/jvs_test_list.txt", "w") as f:
        for fname in jvs_test_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/jvs_eval_list.txt", "w") as f:
        for fname in jvs_eval_file_list:
            f.write("%s\n" % fname)


if __name__ == "__main__":
    create_train_test_list()
