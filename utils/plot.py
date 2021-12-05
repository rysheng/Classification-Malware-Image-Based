import numpy as np
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math


def plot_acc(save_path, acc_train, acc_test):
    save_path = save_path + '/acc.png'

    plt.plot(acc_train, label='train_acc')
    plt.plot(acc_test, label='test_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Accuracy of Train and Test value')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def plot_loss(save_path, loss_train, loss_test):
    save_path = save_path + '/loss.png'

    plt.plot(loss_train, label='train_loss')
    plt.plot(loss_test, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss of Train and Test value')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()