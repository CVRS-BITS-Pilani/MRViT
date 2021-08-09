import torch
import torch.nn as nn
from models.nest import NesT
from models.levit import LeViT

class MultiResViT(nn.Module):
  def __init__(self, nclasses = 4):
    super(MultiResViT, self).__init__()
    self.nclasses = nclasses
    self.small = LeViT(
                        image_size = 112,
                        num_classes = 2, # number of classes
                        stages = 2,             # number of stages
                        dim = (256, 512),  # dimensions at each stage
                        depth = 2,              # transformer depth
                        heads = (4, 8),      # heads at each stage
                        mlp_mult = 2,
                        dropout = 0.10  # dropout rate
                    )
    self.large = LeViT(
                        image_size = 112,
                        num_classes = self.nclasses - 2, # number of classes
                        stages = 2,             # number of stages
                        dim = (256, 512),  # dimensions at each stage
                        depth = 2,              # transformer depth
                        heads = (4, 8),      # heads at each stage
                        mlp_mult = 2,
                        dropout = 0.10  # dropout rate
                    )
    self.fc = nn.Linear(self.nclasses, self.nclasses)  # set up FC layer
  def forward(self, input1, input2):
    s = self.small(input2)
    l = self.large(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((s.view(s.size(0), -1),
                          l.view(l.size(0), -1)), dim=1)
    out = self.fc(combined)
    return out