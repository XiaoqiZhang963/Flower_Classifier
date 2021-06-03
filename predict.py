# ------------ Imports ----------------
import argparse
from PIL import Image
import numpy as np
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# ------------- Get inputs ---------------
parser = argparse.ArgumentParser(description='Get inputs')
# Basic usage
parser.add_argument('image_path', action="store")
parser.add_argument('checkpoint', action="store")
# Options
parser.add_argument('--top_k', action="store", type=int, default=3)
parser.add_argument('--category_names', action="store", default="cat_to_name.json")
parser.add_argument('--gpu', action="store_true")
args = parser.parse_args()

# ------------ Load Checkpoint Function ----------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']

    return model

# ------------ Image Preprocessing Function --------------
def process_image(image):
    im = Image.open(image)

    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    a, b = im.size
    aspect_ratio = a / b
    if a > b:
        im = im.resize((round(aspect_ratio * 256), 256))
    else:
        im = im.resize((256, round(256 / aspect_ratio)))

    # Crop out the center 224x224 portion of the image
    a, b = im.size
    left = round((a - 224)/2)
    top = round((b - 224)/2)
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))

    # Convert color channels to 0-1
    im_np = np.array(im)/255

    # Normalize
    im_np = (im_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # Reorder dimensions
    im_np = im_np.transpose((2, 0, 1))

    im_torch = torch.from_numpy(im_np)

    return im_torch

# --------------- Prediction Function --------------
def predict(image_path, model, topk=3):

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    image = process_image(image_path).unsqueeze_(0).float()
    model, image = model.to(device), image.to(device)
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk, dim=1)

        top_p = top_p.squeeze().tolist()

        idx_to_class = dict((v,k) for k,v in model.class_to_idx.items())
        top_class = [idx_to_class[int(idx)] for idx in top_idx[0]]
    return top_p, top_class

# ---------------- Predict Classes -------------------
path = args.image_path
model = load_checkpoint(args.checkpoint)
topk = args.top_k
cat_to_name = args.category_names

with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(path, model, topk)
names = [cat_to_name[cat] for cat in classes]
for i in range(topk):
    print(names[i],": probability = ", str(round(probs[i]*100))+'%')
