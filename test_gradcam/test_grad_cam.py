import glob

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import vgg16,resnet50
from model.basemodel import MyResNet
import torch
import cv2
import numpy as np
import os
# model = resnet50(pretrained=False)
model = MyResNet(num_classes=2)
checkpoint = torch.load("../resutls/exp_malware_resnet50/best.pt", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
# model.eval()
target_layers = [model.model.layer4[-1]]
# target_layers = [model.features[-1]]




cam = GradCAM(model=model, target_layers=target_layers)

target_category = 281

paths = "G:/PYTHON/useGradCam_RansomAndBenign/check_all/ransome/after_pe/"
for path in glob.glob(paths + "*"):
    rgb_img = cv2.imread(path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # path_save =  "G:/PYTHON/test_ransomAndbenign/check_all/check_cam/before_pe/vgg16/ransome/vgg_"+ path.split('\\')[-1]
    # cv2.imwrite(path_save, visualization)
    # print(path_save)
    cv2.imshow("debug", visualization)
    cv2.waitKey(0)