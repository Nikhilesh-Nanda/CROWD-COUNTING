import cv2
import numpy as np
import os
from ultralytics import YOLO
from scipy.ndimage import maximum_filter

# -------------------
# Paths (WSL)
# -------------------
base_path = "/home/kiit/crowd_detection/YOLO/DEMO/CSRNet_sample_dataset"

img_folder = os.path.join(base_path, "train_img")
density_folder = os.path.join(base_path, "train_gr")
vis_folder = os.path.join(base_path, "visualization")

model_path = "/home/kiit/crowd_detection/YOLO/DEMO/models/weights.pt"

os.makedirs(density_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

# -------------------
# Load YOLO
# -------------------
model = YOLO(model_path)

# -------------------
# Density function
# -------------------
def generate_density_map(image, points, sigma=4):

    h, w = image.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for x, y in points:

        x = int(x)
        y = int(y)

        if x<0 or y<0 or x>=w or y>=h:
            continue

        size = int(6*sigma+1)

        g = cv2.getGaussianKernel(size, sigma)
        g = g @ g.T
        g = g / g.sum()

        x1 = x-size//2
        y1 = y-size//2
        x2 = x1+size
        y2 = y1+size

        gx1 = max(0, -x1)
        gy1 = max(0, -y1)
        gx2 = min(size, w-x1)
        gy2 = min(size, h-y1)

        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(w, x2)
        dy2 = min(h, y2)

        density[dy1:dy2, dx1:dx2] += g[gy1:gy2, gx1:gx2]

    return density


# -------------------
# Process images
# -------------------
for img_name in os.listdir(img_folder):

    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(img_folder, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    # YOLO detection
    results = model(image)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    points = []

    for b in boxes:

        x1,y1,x2,y2 = b[:4]

        cx = (x1+x2)/2
        cy = (y1+y2)/2

        points.append((cx,cy))

    # density map
    density = generate_density_map(image, points, sigma=4)

    density = cv2.resize(density, (w//8, h//8))

    density = density * 64

    # save npy
    density_path = os.path.join(
        density_folder,
        os.path.splitext(img_name)[0] + ".npy"
    )

    np.save(density_path, density)

    # visualization
    density_up = cv2.resize(density, (w,h))

    local_max = maximum_filter(density_up, size=15)

    thresh = density_up.max() * 0.3

    peaks = (density_up == local_max) & (density_up > thresh)

    ys, xs = np.where(peaks)

    vis_img = image.copy()

    for x,y in zip(xs,ys):

        cv2.circle(vis_img, (x,y), 4, (0,0,255), -1)

    # name_op.jpg
    name, ext = os.path.splitext(img_name)

    vis_path = os.path.join(vis_folder, f"{name}_op.jpg")

    cv2.imwrite(vis_path, vis_img)

    print(
        img_name,
        "| heads:", len(points),
        "| density sum:", round(density.sum(),2)
    )

print("done")