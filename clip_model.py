import torch
import clip
from PIL import Image

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device)
