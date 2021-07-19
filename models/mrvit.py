import torch
import torch.nn as nn
from models.nest import NesT
from models.levit import LeViT
from models.cct import cct_2

# self.small = NesT(
  #                     image_size = 112,
  #                     patch_size = 7,
  #                     dim = 49,
  #                     heads = 3,
  #                     num_hierarchies = 3,        # number of hierarchies
  #                     block_repeats = (8, 4, 1),  # the number of transformer blocks at each heirarchy, starting from the bottom
  #                     num_classes = self.nclasses - 1 # number of classes
  #                 )
# self.small = cct_2(
#                     img_size = 224,
#                     n_conv_layers = 1,
#                     kernel_size = 7,
#                     stride = 2,
#                     padding = 3,
#                     pooling_kernel_size = 3,
#                     pooling_stride = 2,
#                     pooling_padding = 1,
#                     num_classes = 1,
#                     positional_embedding='learnable', # ['sine', 'learnable', 'none']  
#                   )

class MultiResViT(nn.Module):
  def __init__(self, nclasses = 4):
    super(MultiResViT, self).__init__()
    self.nclasses = nclasses
    self.small = LeViT(
                        image_size = 224,
                        num_classes = 2,
                        stages = 2,             # number of stages
                        dim = (256, 512),  # dimensions at each stage
                        depth = 2,              # transformer depth
                        heads = (4, 8),      # heads at each stage
                        mlp_mult = 2, 
                        dropout = 0.10  # dropout rate
                    )
    self.large = LeViT(
                        image_size = 224,
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