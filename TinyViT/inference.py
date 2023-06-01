"""Model Inference."""
import torch
import numpy as np
from PIL import Image

from models.tiny_vit import tiny_vit_21m_224
from data import build_transform, imagenet_classnames
from config import get_config

config = get_config()


# Build model
model = tiny_vit_21m_224(pretrained=True)
model.eval()

# Load Image
fname = './.figure/cat.jpg'
image = Image.open(fname)
transform = build_transform(is_train=False, config=config)

# (1, 3, img_size, img_size)
batch = transform(image)[None]

with torch.no_grad():
    logits = model(batch)

# print top-5 classification names
probs = torch.softmax(logits, -1)
scores, inds = probs.topk(5, largest=True, sorted=True)
print('=' * 30)
print(fname)
for score, ind in zip(scores[0].numpy(), inds[0].numpy()):
    print(f'{imagenet_classnames[ind]}: {score:.2f}')
