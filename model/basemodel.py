import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.decomposition import PCA
import torch
import timm


class IMCEC_origin(nn.Module):
    def __init__(self, num_classes):
        super(IMCEC_origin, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.vgg16 = models.vgg16(pretrained=True)

        self.pca_hstack_resnet50 = PCA(n_components=205)
        self.pca_hstack_vgg16 = PCA(n_components=410)
        self.pca_hstack = PCA(n_components=615)

        self.resnet.fc = nn.Sequential(nn.Flatten())

        self.vgg16.classifier = nn.Sequential(nn.Flatten(),
                                              nn.Linear(25088, 4096))

        self.fc_resnet = nn.Sequential(nn.Linear(2048, 1024),
                                       nn.ReLU(True),
                                       nn.Dropout(p=0.65),
                                       nn.Linear(1024, num_classes))

        self.fc_vgg16 = nn.Sequential(nn.Linear(4096, 1024),
                                      nn.ReLU(True),
                                      nn.Dropout(p=0.65),
                                      nn.Linear(1024, num_classes))

        self.linear_hstack_restnet50 = nn.Linear(205, num_classes)
        self.linear_hstack_vgg16 = nn.Linear(410, num_classes)
        self.linear_hstack = nn.Linear(615, num_classes)

    def forward(self, x):
        input_resnet50 = self.resnet(x)
        input_vgg16 = self.vgg16(x)

        hstack_resnet50 = self.pca_hstack_resnet50.fit_transform(input_resnet50.detach().numpy())
        hstack_resnet50 = self.linear_hstack_restnet50(torch.from_numpy(hstack_resnet50))

        hstack_vgg16 = self.pca_hstack_vgg16.fit_transform(input_vgg16.detach().numpy())
        hstack_vgg16 = self.linear_hstack_vgg16(torch.from_numpy(hstack_vgg16))

        hstack = self.pca_hstack.fit_transform((torch.cat((input_resnet50, input_vgg16), dim=1)).detach().numpy())
        hstack = self.linear_hstack(torch.from_numpy(hstack))

        fc_resnet = self.fc_resnet(input_resnet50)
        fc_vgg16 = self.fc_vgg16(input_vgg16)

        output = torch.sum(torch.stack((hstack_vgg16, hstack_resnet50, hstack, fc_resnet, fc_vgg16)), dim=0)
        output = F.softmax(output, dim=1)
        return output


class IMCEC(nn.Module):
    def __init__(self, num_classes):
        super(IMCEC, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.vgg16 = models.vgg16(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.pca_hstack_resnet50 = nn.Linear(2048, num_classes)
        self.pca_hstack_vgg16 = nn.Linear(4096, num_classes)
        self.pca_hstack = nn.Linear(6144, num_classes)

        self.resnet.fc = nn.Sequential(nn.Flatten())

        self.vgg16.classifier = nn.Sequential(nn.Linear(25088, 4096))

        self.fc_resnet = nn.Sequential(nn.Linear(2048, 1024),
                                       nn.Softmax(dim=1),
                                       nn.Dropout(p=0.65),
                                       nn.Linear(1024, num_classes))

        self.fc_vgg16 = nn.Sequential(nn.Linear(4096, 1024),
                                      nn.Softmax(dim=1),
                                      nn.Dropout(p=0.65),
                                      nn.Linear(1024, num_classes))

        self.fc = nn.Sequential(nn.Linear(5 * num_classes, 1024),
                                nn.Softmax(dim=1),
                                nn.Dropout(p=0.65),
                                nn.Linear(1024, num_classes))

    def forward(self, x):
        input_resnet50 = self.resnet(x)
        input_vgg16 = self.vgg16(x)

        hstack_resnet50 = self.pca_hstack_resnet50(input_resnet50)

        hstack_vgg16 = self.pca_hstack_vgg16(input_vgg16)

        hstack = self.pca_hstack(torch.cat((input_resnet50, input_vgg16), dim=1))

        resnet_fc = self.fc_resnet(input_resnet50)
        vgg16_fc = self.fc_vgg16(input_vgg16)

        output = self.fc(torch.cat((hstack_vgg16, hstack_resnet50, hstack, resnet_fc, vgg16_fc), dim=1))
        output = F.normalize(output, p=2, dim=1)
        return output


class MyResNet(nn.Module):
    def __init__(self, num_classes):
        super(MyResNet, self).__init__()

        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1)
        output = F.normalize(output, p=2, dim=1)
        return output


class MyVgg16(nn.Module):
    def __init__(self, num_classes):
        super(MyVgg16, self).__init__()

        self.model = models.vgg16(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        output = output.view(output.size(0), -1)
        output = F.normalize(output, p=2, dim=1)
        return output


class ViTranformer(nn.Module):
    def __init__(self, num_classes):
        super(ViTranformer, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.Softmax(dim=1),
            nn.Dropout(p=0.65),
            nn.Linear(512, num_classes))

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        output = output.view(output.size(0), -1)
        output = F.normalize(output, p=2, dim=1)
        return output
