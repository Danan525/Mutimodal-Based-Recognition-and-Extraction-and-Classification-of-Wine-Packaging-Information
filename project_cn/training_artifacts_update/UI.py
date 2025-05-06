import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ================= 模型结构 ==================
class DynamicDualBranchMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.pack_net = nn.Sequential(
            *list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gcms_net = nn.Sequential(
            *list(models.resnet18(weights='IMAGENET1K_V1').children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pack_imgs, gcms_imgs):
        pack_imgs = torch.stack(pack_imgs)
        gcms_imgs = torch.stack(gcms_imgs)
        pack_feats = self.pack_net(pack_imgs).view(pack_imgs.size(0), -1)
        gcms_feats = self.gcms_net(gcms_imgs).view(gcms_imgs.size(0), -1)
        fused = torch.cat([pack_feats, gcms_feats], dim=1)
        return self.classifier(fused).squeeze()

# ================= 模型加载 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicDualBranchMatcher().to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

# ================= 图像预处理 ==================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= 推理函数 ==================
def predict(pack_img, gcms_img):
    with torch.no_grad():
        pack_tensor = preprocess(pack_img).to(device)
        gcms_tensor = preprocess(gcms_img).to(device)

        output = model([pack_tensor], [gcms_tensor])
        prob = output.item()

        if prob >= 0.5:
            return f"✅ 匹配成功（真酒）\n置信度：{prob:.4f}"
        else:
            return f"❌ 不匹配（疑似假酒）\n置信度：{prob:.4f}"

# ================= UI 界面 ==================
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="上传包装图像"),
        gr.Image(type="pil", label="上传 GC-MS 图像")
    ],
    outputs="text",
    title="白酒真伪判定系统",
    description="上传一张包装图和一张 GC-MS 图，模型将判断是否为匹配的真酒对。"
)

iface.launch()
