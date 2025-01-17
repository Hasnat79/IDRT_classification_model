import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
class BuildingDataset(Dataset):

  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    # self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    # self.classes = ['cat07 MasAptMotel', 'cat10 StripMall', 'cat11 L_ShopMall', 'cat17 LowRise', 'cat18 MidRise', 'cat19 HighRise', 'cat20 InstiBldg', 'cat21 MetalBldg', 'cat22 Canopy']
    self.classes = [  'cat10 StripMall', 'cat11 L_ShopMall',  'cat18 MidRise', 'cat19 HighRise', 'cat21 MetalBldg', 'cat22 Canopy']
    self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    self.images = []
    self.labels = []
    
    for class_name in self.classes:
        class_dir = os.path.join(root_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.jpg'):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
      img_path = self.images[idx]
      image = Image.open(img_path).convert('RGB')
      label = self.labels[idx]
      
      if self.transform:
          image = self.transform(image)
          
      return image, label

  # generate class distribution of images
  def class_distribution(self):
    class_dist = {cls: 0 for cls in self.classes}
    for label in self.labels:
        class_dist[self.classes[label]] += 1
    return class_dist
  # plot class distribution of images
  def plot_class_distribution(self, path=None):
    class_dist = self.class_distribution()
    plt.figure(figsize=(10, 5))
    plt.bar(class_dist.keys(), class_dist.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Added to prevent label cutoff
    plt.savefig(path)
  def get_dataset_details(self):
    print(f"Number of images: {len(self)}")
    print(f"Number of classes: {len(self.classes)}")
    print(f"Classes: {self.classes}")
    print(f"class distribution: {self.class_distribution()}")
