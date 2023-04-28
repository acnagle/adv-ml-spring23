import matplotlib.pyplot as plt
import torch

ckpt_dict = {
    'train': ['./results/train/ckpt_ep=100_lr=0.01.pt', './results/train/ckpt_ep=100_lr=0.01_augment.pt'],
    'train2': ['./results/train2/ckpt_ep=100_lr=0.01.pt', './results/train2/ckpt_ep=100_lr=0.01_augment.pt'],
    'lora_train': ['./results/lora_train/ckpt_ep=100_lr=0.01.pt', './results/lora_train/ckpt_ep=100_lr=0.01_augment.pt'],
    'pti_train': ['./results/pti_train/ckpt_ep=100_lr=0.01.pt', './results/pti_train/ckpt_ep=100_lr=0.01_augment.pt'],
}

legend_dict = {
    'train': ['base-5', 'base-5-aug'],
    'train2': ['base-10', 'base-10-aug'],
    'lora_train': ['sd-lora', 'sd-lora-aug'],
    'pti_train': ['sd-lti', 'sd-lti-aug'],
}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for ckpt_key in ckpt_dict:
    for i in range(len(ckpt_dict[ckpt_key])):
        ckpt = torch.load(ckpt_dict[ckpt_key][i])

        ax[0].plot(ckpt['stats']['epoch'], ckpt['stats']['train_acc'], linewidth=2, label=legend_dict[ckpt_key][i])
        ax[1].plot(ckpt['stats']['epoch'], ckpt['stats']['val_acc'], linewidth=2)

ax[0].set_xlabel('Epoch', fontsize=16)
ax[0].set_ylabel('Training Accuracy', fontsize=16)
ax[1].set_xlabel('Epoch', fontsize=16)
ax[1].set_ylabel('Test Accuracy', fontsize=16)
ax[0].legend(fontsize=12)
ax[0].grid(True)
ax[1].grid(True)

plt.savefig('./results/bmw10_plot.png', bbox_inches='tight', dpi=300)
