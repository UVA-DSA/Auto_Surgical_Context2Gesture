import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch


#weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
def Deeplabv3_rob(pretrained=True, num_classes=1): # background, grsper, needle, thread
    model = models.segmentation.deeplabv3_resnet50( pretrained=pretrained,progress=True)
    model.aux_classifier = None
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = DeepLabHead(2048, num_classes)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    return model