import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ================= Model Definition ==================
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

# ================= Model Loading ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicDualBranchMatcher().to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

# ================= Image Preprocessing ==================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= Inference Function ==================
def predict(pack_img, gcms_img):
    with torch.no_grad():
        pack_tensor = preprocess(pack_img).to(device)
        gcms_tensor = preprocess(gcms_img).to(device)

        output = model([pack_tensor], [gcms_tensor])
        prob = output.item()

        if prob >= 0.5:
            return f"✅ Match Found (Real)\nConfidence: {prob:.4f}"
        else:
            return f"❌ No Match (Fake)\nConfidence: {prob:.4f}"

# ================= Custom CSS for Big Fonts ==================
custom_css = """
h1, h2, h3, label, button, .output-textbox, .input-image {
    font-size: 36px !important;
}

.gr-button {
    font-size: 36px !important;
    padding: 20px 40px !important;
}

textarea, .output-textbox {
    font-size: 36px !important;
}
"""

# ================= Gradio Interface ==================
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("<h1 style='text-align: center;'>Baijiu Authenticity Verification System</h1>")
    gr.Markdown("<h2 style='text-align: center;'>Upload a packaging image and a GC-MS spectrum. The model will determine whether they match.</h2>")
    
    with gr.Row():
        pack_input = gr.Image(type="pil", label="Upload Packaging Image")
        gcms_input = gr.Image(type="pil", label="Upload GC-MS Image")
    
    output_text = gr.Textbox(label="Prediction Result")

    run_button = gr.Button("Run Prediction")

    run_button.click(fn=predict, inputs=[pack_input, gcms_input], outputs=output_text)

iface.launch()
