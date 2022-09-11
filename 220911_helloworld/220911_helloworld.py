import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import kornia as K

# Load an image with OpenCV
img_bgr: np.array = cv2.imread('arturito.jpg')  # HxWxC / np.uint8
# matplotlib의 이미지 출력 형식이 BGR이므로 변환
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
#plt.show()

# Load an image with Torchvision
# 이미지를 torch.Tensor로 받아옴(C,H,W)
x_rgb: torch.tensor = torchvision.io.read_image('arturito.jpg')  # CxHxW / torch.uint8
x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
print(x_rgb.shape)

# Load an image with Kornia
# image_to_tensor(): numpy array -> tensor
x_bgr: torch.tensor = K.image_to_tensor(img_bgr)  # CxHxW / torch.uint8
x_bgr = x_bgr.unsqueeze(0)  # 1xCxHxW
print(f"convert from '{img_bgr.shape}' to '{x_bgr.shape}'")
