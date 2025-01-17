import torch
import torch.nn as nn
import timm


class ViTFineTune(nn.Module):
  def __init__(self, num_classes, pretrained=True):
    super(ViTFineTune, self).__init__()

    # fetch the pretrained ViT model
    self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, cache_dir="cache")

    # change the head to output num_classes (default is 1000 for imagened1k)
    self.vit.head = nn.Sequential(
      nn.Dropout(p=0.5), # adding dropout to tackle overfitting
      nn.Linear(self.vit.head.in_features, num_classes)
    )
    # print(self.vit.head.in_features)

    # initially freeze all layers
    for param in self.vit.parameters():
      param.requires_grad = False
    
    # unfreeze last 5 out of 12 transformer blocks
    for param in self.vit.blocks[-5:].parameters():
      param.requires_grad = True
    
    # unfreeze the head
    for param in self.vit.head.parameters():
      param.requires_grad = True
  
  def forward(self, x):
    return self.vit(x)
  
if __name__ == "__main__":
  model = ViTFineTune(6)
  print(model)


