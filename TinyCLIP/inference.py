import torch
from PIL import Image
import open_clip

# manual inheritance
# arch = 'TinyCLIP-ViT-39M-16-Text-19M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='YFCC15M')

# arch = 'TinyCLIP-ViT-8M-16-Text-3M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='YFCC15M')

# arch = 'TinyCLIP-ResNet-30M-Text-29M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

# arch = 'TinyCLIP-ResNet-19M-Text-19M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

# arch = 'TinyCLIP-ViT-61M-32-Text-29M'
# model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

arch = 'TinyCLIP-ViT-40M-32-Text-19M'
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='LAION400M')

tokenizer = open_clip.get_tokenizer(arch)

image_fname = './figure/TinyCLIP.jpg'
image = preprocess(Image.open(image_fname)).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
