import torch.nn as nn
from torchvision import models

def resnet18(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    return FeatureExtractor(model)

def resnet34(pretrained=True):
    model = models.resnet34(pretrained=pretrained)
    return FeatureExtractor(model)

def resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return FeatureExtractor(model)

def resnet101(pretrained=True):
    model = models.resnet101(pretrained=pretrained)
    return FeatureExtractor(model)

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.features(x)
        f = x
        class_f = self.avgpool(x)
        class_f = class_f.view(class_f.size(0), -1)
        # We don't necessarily need the final FC here for PSPNet if it uses its own classifier, 
        # but PSPNet's forward expects (f, class_f)
        return f, class_f
