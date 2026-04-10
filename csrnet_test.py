import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# Paths (WSL)
# -------------------------
model_path = "/home/kiit/crowd_detection/YOLO/DEMO/models/csrnet_epoch_180.pth"

input_folder = "/home/kiit/crowd_detection/YOLO/DEMO/test_img/csrnet_test"

# save in same folder
output_folder = input_folder

os.makedirs(output_folder, exist_ok=True)

# ==============================
# CSRNet Model (UNCHANGED)
# ==============================
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        self.frontend_feat = [
            64,64,'M',
            128,128,'M',
            256,256,256,'M',
            512,512,512
        ]

        self.backend_feat = [
            512,512,512,256,128,64
        ]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(
            self.backend_feat,
            in_channels=512,
            dilation=True
        )

        self.output_layer = nn.Conv2d(
            64,1,kernel_size=1
        )

    def forward(self,x):

        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        return x


    def _make_layers(
        self,
        cfg,
        in_channels=3,
        dilation=False
    ):

        layers = []

        for v in cfg:

            if v == 'M':

                layers.append(
                    nn.MaxPool2d(2,2)
                )

            else:

                d_rate = 2 if dilation else 1

                layers.append(

                    nn.Conv2d(
                        in_channels,
                        v,
                        3,
                        padding=d_rate,
                        dilation=d_rate
                    )

                )

                layers.append(
                    nn.ReLU(inplace=True)
                )

                in_channels = v

        return nn.Sequential(*layers)


# ==============================
# Device
# ==============================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==============================
# Load Model
# ==============================
model = CSRNet().to(device)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("model loaded")


# ==============================
# preprocessing
# ==============================
transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]

    )

])


# ==============================
# process images
# ==============================
for file in os.listdir(input_folder):

    if not file.lower().endswith(
        (".jpg",".png",".jpeg")
    ):
        continue

    if file.endswith("_op.jpg"):
        continue


    img_path = os.path.join(
        input_folder,
        file
    )

    img = Image.open(
        img_path
    ).convert("RGB")

    w,h = img.size


    input_tensor = transform(img)\
        .unsqueeze(0)\
        .to(device)


    with torch.no_grad():

        density = model(
            input_tensor
        )


    density = density.squeeze()\
        .cpu()\
        .numpy()


    count = density.sum()

    print(
        file,
        "->",
        round(count,2)
    )


    # resize for visualization
    density_resized = cv2.resize(
        density,
        (w,h)
    )


    # heatmap
    density_norm = density_resized / (
        density_resized.max()+1e-6
    )

    heatmap = cv2.applyColorMap(

        np.uint8(
            255*density_norm
        ),

        cv2.COLORMAP_JET

    )


    original = cv2.imread(
        img_path
    )


    overlay = cv2.addWeighted(

        original,
        0.6,

        heatmap,
        0.4,

        0

    )


    cv2.putText(

        overlay,

        f"Count: {int(count)}",

        (20,40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0,255,0),

        2

    )


    name,_ = os.path.splitext(file)

    save_path = os.path.join(

        output_folder,

        f"{name}_op.jpg"

    )


    cv2.imwrite(

        save_path,

        overlay

    )


print("done")