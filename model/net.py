import torch
from model import basemodel
import os


def get_model(args, device):
    if args.model == 'IMCEC':
        model = basemodel.IMCEC(args.num_class)
    elif args.model == 'Resnet50':
        model = basemodel.MyResNet(args.num_class)
    elif args.model == 'Vgg16':
        model = basemodel.MyVgg16(args.num_class)
    elif args.model == 'ViT':
        model = basemodel.ViTranformer(args.num_class)

    model = model.to(device)
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp, map_location=torch.device(device))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))

    return model