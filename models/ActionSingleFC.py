import torch.nn as nn
import torch.nn.functional as F

class ActionSingleFC(nn.Module):
    def __init__(self, num_classes, feature_dim=1024) -> None:
        self.feature_dim = feature_dim
        self.num_classes = num_classes


        self.fc = nn.Linear(input_size=self.feature_dim, output_size=self.num_classes)


    def forward(self, x):
        x = F.relu(self.fc(x))
        return x