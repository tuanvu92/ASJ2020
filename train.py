import matplotlib
matplotlib.use("Agg")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
from data_utils.mel_dataset import *
import os
from os.path import join, exists
from time import localtime, strftime
from torch.utils.data.distributed import DistributedSampler
from progressbar import progressbar
from utils.logger import DataLogger
from utils.validator import Validator
from torch.optim import Adam
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from models.vqvae_3stage import VectorQuantisedVAE as vqvae_3stage
from models.vqvae_2stage import VectorQuantisedVAE as vqvae_2stage
from models.vqvae_1stage import VectorQuantisedVAE as vqvae_1stage


def train_crosslingual(num_gpus=1, rank=0, group_name="", n_stage=1, n_codebook=1):
    if num_gpus > 1:
        init_distributed(rank=rank, num_gpus=num_gpus, group_name=group_name,
                         dist_backend="nccl", dist_url="tcp://150.65.248.54:54321")
    timestamp = strftime("%Y%m%d_%H%M_vqvae", localtime())
    timestamp = timestamp + "_%d_stage_%d_cb" % (n_stage, n_codebook)
    checkpoint_path = join("/home/messier/PycharmProjects/ASJ2020/checkpoints/", timestamp)
    if rank == 0:
        print("Output dir: %s" % checkpoint_path)
        if not exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    en_train_list = 'file_lists/vctk_train_list.txt'
    jp_train_list = 'file_lists/jvs_train_list.txt'
    dataset = CrossLingualDataset(en_train_list, jp_train_list)
    with open(join(checkpoint_path, "speaker_label.json"), "w") as f:
        speaker_label = dict()
        speaker_label.update(dataset.speaker_label_en)
        for k, v in dataset.speaker_label_jp.items():
            speaker_label[k] = v + len(dataset.speaker_label_en)
        json.dump(speaker_label, f)

    print("Dataset start_index: ", dataset.start_idx)
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    print("No. training data: ", len(dataset.mel_file_list_en) + len(dataset.mel_file_list_jp))

    collate_fn = CrossLingualCollateFn(max_seq_len=512)
    dataloader = DataLoader(dataset=dataset,
                            sampler=train_sampler,
                            pin_memory=True,
                            batch_size=16,
                            collate_fn=collate_fn,
                            num_workers=4,
                            shuffle=False)
    if n_stage == 1:
        model = vqvae_1stage(n_speaker=dataset.n_speaker,
                             n_codebook=n_codebook,
                             codebook_size_bot=384,
                             downsampling_bot=2).cuda()
    elif n_stage == 2:
        model = vqvae_2stage(n_speaker=dataset.n_speaker,
                             n_codebook=n_codebook,
                             codebook_size_bot=192, codebook_size_top=192,
                             downsampling_top=4, downsampling_bot=2).cuda()
    elif n_stage == 3:
        model = vqvae_3stage(n_speaker=dataset.n_speaker,
                             n_codebook=n_codebook,
                             codebook_size_bot=128, codebook_size_mid=128, codebook_size_top=128,
                             downsampling_top=2, downsampling_mid=2, downsampling_bot=2).cuda()
    else:
        print("n_stage > 3")
        exit()
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======

    lr = 2e-4
    optimizer = Adam(model.parameters(), lr=lr)
    logger = None
    validator = None
    if rank == 0:
        logger = DataLogger(logdir=join(checkpoint_path, "logs"))
        validator = Validator(logger, n_speaker=dataset.n_speaker, target_id=0)

    iteration = 0
    for epoch in range(1000):
        model.train()
        # Need to call this to shuffle data every epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            iterator = progressbar(dataloader, redirect_stdout=True)
        else:
            iterator = dataloader

        for batch in iterator:
            batch_size = batch[0].shape[0]
            mcc_en = batch[0][:batch_size//2].cuda()
            mcc_jp = batch[0][batch_size//2:].cuda()
            y_en = batch[1][:batch_size//2].cuda()
            y_jp = batch[1][batch_size//2:].cuda()
            language_id_en = torch.zeros([mcc_en.shape[0], 2], dtype=torch.float32).cuda()
            language_id_en[:, 0] = 1.
            language_id_jp = torch.zeros([mcc_jp.shape[0], 2], dtype=torch.float32).cuda()
            language_id_jp[:, 1] = 1.

            model.zero_grad()
            loss_en, loss_components_en = model([mcc_en, y_en, language_id_en])
            loss_jp, loss_components_jp = model([mcc_jp, y_jp, language_id_jp])
            loss = loss_en + loss_jp
            loss.backward()
            optimizer.step()

            if num_gpus > 1:
                reduced_loss_en = reduce_tensor(loss_en.data, num_gpus).item()
                for i in range(len(loss_components_en)):
                    loss_components_en[i] = reduce_tensor(loss_components_en[i].data, num_gpus).item()
                reduced_loss_jp = reduce_tensor(loss_jp.data, num_gpus).item()
                for i in range(len(loss_components_jp)):
                    loss_components_jp[i] = reduce_tensor(loss_components_jp[i].data, num_gpus).item()
            else:
                reduced_loss_en = loss_en.item()
                for i in range(len(loss_components_en)):
                    loss_components_en[i] = loss_components_en[i].item()
                reduced_loss_jp = loss_jp.item()
                for i in range(len(loss_components_jp)):
                    loss_components_jp[i] = loss_components_jp[i].item()

            rc_loss, mel_loss, vq_loss, commitment_loss, \
                perplexity_top, perplexity_mid, perplexity_bot = loss_components_en

            rc_loss_jp, mel_loss_jp, vq_loss_jp, commitment_loss_jp, \
                perplexity_top_jp, perplexity_mid_jp, perplexity_bot_jp = loss_components_jp

            print("[%d|%d]: loss=[%.2e|%.2e], "
                  "rc_loss=[%.2e|%.2e], mel_loss=%.2e, vq_loss=%.2e, "
                  "perplexity_en=[%.2e|%.2e|%.2e], perplexity_jp=[%.2e|%.2e|%.2e] " %
                  (epoch, iteration,
                   reduced_loss_en, reduced_loss_jp,
                   rc_loss, rc_loss_jp,
                   mel_loss,
                   vq_loss,
                   perplexity_top,
                   perplexity_mid,
                   perplexity_bot,
                   perplexity_top_jp,
                   perplexity_mid_jp,
                   perplexity_bot_jp,
                   ))

            # Log data to tensorboard
            if rank == 0:
                logger.log_training([loss,
                                     rc_loss, mel_loss, vq_loss,
                                     rc_loss_jp, mel_loss_jp, vq_loss_jp,
                                     [perplexity_top, perplexity_mid, perplexity_bot],
                                     [perplexity_top_jp, perplexity_mid_jp, perplexity_bot_jp]
                                     ],
                                    ["loss",
                                     "English/rc_loss", "English/mel_loss", "English/vq_loss",
                                     "Japanese/rc_loss", "Japanese/mel_loss", "Japanese/vq_loss",
                                     ["Perplexities_en", "top", "mid", "bot"],
                                     ["Perplexities_jp", "top", "mid", "bot"]],
                                    iteration)
                if iteration % 1000 == 0:
                    validator(model, iteration)
                    torch.save(model.state_dict(),
                               join(checkpoint_path, "vqvae_en_jp_weight_latest.pt"))
                    if iteration % 10000 == 0:
                        torch.save(model.state_dict(),
                                   join(checkpoint_path, "vqvae_en_jp_weight_{}.pt".format(iteration)))
            iteration += 1

            if iteration >= 50000:
                print("Finished")
                exit()

        dataset.random_start_idx()
        if epoch < 5:
            lr = max(2e-4, lr - 1e-4)
        else:
            lr = max(1e-5, lr - 1e-6)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-m', '--mode', type=int, default=1,
                        help='run mode. 0: Inference, 1: train vqvae_1stage, '
                             '2: train_vqvae_2stage, 3: train_vqvae_3stage')
    parser.add_argument('-c', '--checkpoint', type=str, default='',
                        help='path to checkpoint')
    parser.add_argument('-n', '--norm', type=bool, default=False, help='Normalize mcc')
    parser.add_argument('-s', '--speaker_label', type=str, default='', help='speaker label file')
    parser.add_argument('-d', '--dst_dir', type=str, default='', help='output dir')
    parser.add_argument('-l', '--mel_dir', type=str, default='', help='source mel dir')
    parser.add_argument('-a', '--language', type=str, default='german', help='language')
    args = parser.parse_args()

    if args.mode == 1:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if args.group_name == '':
                print("Warning: Training on 1 GPU!")
                num_gpus = 1
        train_crosslingual(num_gpus, args.rank, args.group_name, n_stage=1)

    if args.mode == 2:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if args.group_name == '':
                print("Warning: Training on 1 GPU!")
                num_gpus = 1

        train_crosslingual(num_gpus, args.rank, args.group_name, n_stage=2)

    if args.mode == 3:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if args.group_name == '':
                print("Warning: Training on 1 GPU!")
                num_gpus = 1

        train_crosslingual(num_gpus, args.rank, args.group_name, n_stage=3, n_codebook=1)

    if args.mode == 4:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if args.group_name == '':
                print("Warning: Training on 1 GPU!")
                num_gpus = 1
        train_crosslingual(num_gpus, args.rank, args.group_name, n_stage=3, n_codebook=2)
