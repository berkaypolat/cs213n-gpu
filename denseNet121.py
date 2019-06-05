import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, out_size),
            nn.Sigmoid(),
        )
        
        self.confidence = nn.Linear(num_features, 1)
        nn.init.constant_(self.confidence.bias, 0)
        

    def forward(self, x):
        pred = self.densenet121(x)
        
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        conf = self.confidence(out)
        
        return pred, conf
