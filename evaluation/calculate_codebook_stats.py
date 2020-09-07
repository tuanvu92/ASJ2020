import torch
from data_utils.mel_dataset import CollateNoTruncate, VqVaeDataset
from torch.utils.data import Dataset, DataLoader
from progressbar import progressbar


def calculate_codebook_stats(model):
    data_path = "/home/messier/PycharmProjects/data/VCTK/mel24k"
    dataset = VqVaeDataset(data_path, norm=False)
    with open('train_file_list.txt', 'w') as f:
        f.writelines([s + "\n" for s in dataset.mel_file_list])
    f.close()

    collate_fn = CollateNoTruncate()
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, num_workers=4)
    language_id = torch.tensor(0, dtype=torch.long)
    f = open("gen/vqvae/z_id.dat", "w")
    for batch in progressbar(dataloader):
        seq_lengths = batch[2]
        data = [batch[0].cuda(), batch[1].cuda(), language_id]
        _, z_id_top, _ = model.inference(data, return_zid=True)
        z_id_top = z_id_top.cpu().numpy()
        for (_z_id, length) in zip(z_id_top, seq_lengths):
            _z_id = _z_id[:length]
            s = " ".join(str(c) for c in _z_id) + "\n"
            f.write(s)
    f.close()