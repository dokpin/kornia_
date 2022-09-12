import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def mat2torch(frame):
    img = K.image_to_tensor(frame, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

#capture
fname1 = 'kn_church-2.jpg'
fname2 = 'kn_church-8.jpg'



vid = cv2.VideoCapture(0)
matcher = KF.LoFTR(pretrained='outdoor')

fn = 0
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if fn == 0:
        pre_frame = frame
        fn += 1
        continue

    img1 = mat2torch(pre_frame)
    img2 = mat2torch(frame)

    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
              "image1": K.color.rgb_to_grayscale(img2)}
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

    img_bgr = frame
    for i in range(len(mkpts0)):
        cv2.line(img_bgr, (int(mkpts0[i][0]), int(mkpts0[i][1])), (int(mkpts1[i][0]), int(mkpts1[i][1])), (0, 255, 0), thickness=1, lineType=8)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    pre_frame = frame
    fn += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# matcher = KF.LoFTR(pretrained='outdoor')

# input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
#               "image1": K.color.rgb_to_grayscale(img2)}

# with torch.inference_mode():
#     correspondences = matcher(input_dict)

# for k,v in correspondences.items():
#     print (k)

# mkpts0 = correspondences['keypoints0'].cpu().numpy()
# #print(mkpts0[0][0])
# mkpts1 = correspondences['keypoints1'].cpu().numpy()
# #print(len(mkpts1))
# Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
# #print(Fm)
# inliers = inliers > 0



# img_bgr = cv2.imread('kn_church-8.jpg')  # HxWxC / np.uint8
# for i in range(len(mkpts0)):
#     cv2.line(img_bgr, (int(mkpts0[i][0]), int(mkpts0[i][1])), (int(mkpts1[i][0]), int(mkpts1[i][1])), (0, 255, 0), thickness=1, lineType=8)
# cv2.imshow('imb', img_bgr)
# cv2.waitKey()