from torchvision.models import efficientnet_v2_m
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class EfficientV2M(nn.Module):
    def __init__(self):
        super().__init__()
        
        # init weights
        pretrained_effV2M_weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        
        # Load effV2M model
        self.model = efficientnet_v2_m(weights=pretrained_effV2M_weights)
        
        # saving emedding length
        self.embedd_len = self.model.classifier[1].in_features
    
        # change the last fc layer to have an output of 1 for our binary BM task
        self.model.classifier[1] = nn.Linear(self.embedd_len,1,bias=True)
        
        
    def forward(self, image, prints=False):    
        
        out = self.model(image)
    
        return out