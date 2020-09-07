#!/usr/bin/sh

# Generate converted samples
echo "Generate converted samples..."
echo "Vanilla VQVAE"
python inference.py --mode=1 --checkpoint="checkpoints/20200714_0800_vqvae_1_stage_1_cb/vqvae_en_jp_weight_40000.pt" --speaker_label="checkpoints/20200714_0800_vqvae_1_stage_1_cb/speaker_label.json"
echo "Hierarchical VQVAE"
python inference.py --mode=2 --checkpoint="checkpoints/20200714_0805_vqvae_2_stage_1_cb/vqvae_en_jp_weight_40000.pt" \
--speaker_label="checkpoints/20200714_0805_vqvae_2_stage_1_cb/speaker_label.json"

# Plot modulation spectrum
python ./evaluation/objective.py