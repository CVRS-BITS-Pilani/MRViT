import torch
import torch.nn.functional as F
import torch.nn as nn
from models.nest import NesT
from models.levit import LeViT

class MultiResViT(nn.Module):
  def __init__(self, nclasses = 4):
    super(MultiResViT, self).__init__()
    self.nclasses = nclasses
    self.outclasses = nclasses ** 2
    self.small = NesT(
                        image_size = 28,
                        patch_size = 7,
                        dim = 49,
                        heads = 3,
                        num_hierarchies = 2,        # number of hierarchies
                        block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
                        num_classes = self.outclasses
                    )  
    self.large = LeViT(
                        image_size = 112,
                        num_classes = self.outclasses,
                        stages = 3,             # number of stages
                        dim = (256, 384, 512),  # dimensions at each stage
                        depth = 3,              # transformer of depth 3 at each stage
                        heads = (4, 6, 8),      # heads at each stage
                        mlp_mult = 3,          # multiplier of the number of features in the classifier
                        dropout = 0.05
                    )
    self.fc = nn.Linear(self.outclasses, self.nclasses)  # set up FC layer
    # self.act = nn.Softmax(dim=-1)
  def forward(self, input1, input2):
    s = self.small(input1)
    l = self.large(input2)
    out = self.fc(s + l)
    # now we can reshape `c` and `f` to 2D and concat them
    # combined = torch.cat((s.view(s.size(0), -1),
    #                       l.view(l.size(0), -1)), dim=1)
    # out = self.fc(combined)
    return out