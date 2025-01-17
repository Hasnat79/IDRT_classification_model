import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import BuildingDataset
from models.baseline import BaselineCNNCLassifier 
from tqdm import tqdm


## constants
TRAIN_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/train" # 5591 images
VAL_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/validation" # 859 images
EPOCHS = 5

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5, checkpoint_path="../checkpoints/baseline_cnn.pth"):
    
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for each epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with current loss and accuracy
            epoch_loss = running_loss / (total/labels.size(0))  # current average loss
            epoch_acc = 100 * correct / total  # current accuracy
            pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'accuracy': f'{epoch_acc:.2f}%'})
        
        
        # save checkpooint of the lowest epoch loss
        if checkpoint_path and epoch_loss < 0.1:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint_epoch{epoch+1}_loss{epoch_loss:.4f}_acc{epoch_acc:.2f}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1} with loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2f}%")
        print(f"Epoch {epoch+1}/{num_epochs} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

def evaluate_model(model, val_loader, device, classes, confusion_matrix_path=""): 
   
   model.eval()
   all_preds = []
   all_labels = []
   with torch.no_grad(): 
    correct= 0
    total = 0
    pbar = tqdm(val_loader, desc='Evaluating')
    for inputs, labels in pbar:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = outputs.max(1)

      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
      
      # Update progress bar with current accuracy
      acc = 100 * correct / total
      pbar.set_postfix({'accuracy': f'{acc:.2f}%'})
    
   accuracy = 100.0 * correct / total
   print(f"Validation accuracy: {accuracy:.2f}%")

   # confusion matrix
   cm = confusion_matrix(all_labels, all_preds)
   plt.figure(figsize=(10, 10))
   sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
   plt.title('Confusion Matrix')
   plt.xlabel('Predicted Label')
   plt.ylabel('Ground Truth Label')   
   plt.xticks(rotation=45)
   plt.tight_layout()  # Added to prevent label cutoff
   plt.savefig(confusion_matrix_path)
   print("Confusion matrix saved")

   print('\nClassification Report:')
   print(classification_report(all_labels, all_preds, target_names=classes))
   return accuracy, all_preds, all_labels
   

def main(): 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  # baseline transformation 
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])

  train_dataset = BuildingDataset(TRAIN_DIR, transform=transform)
  val_dataset = BuildingDataset(VAL_DIR, transform=transform)

  # ----uncomment to experimentally cut the train_dataset to 200 samples -----
#   train_dataset.images = train_dataset.images[:1500]
#   train_dataset.labels = train_dataset.labels[:1500]
#   val_dataset.images = val_dataset.images[:1500]
#   val_dataset.labels = val_dataset.labels[:1500]
  # ---- uncomment to print dataset details ----
  print(f"===== Train Dataset details=====")
  train_dataset.get_dataset_details()
  # print(f"===== Validation Dataset details=====")
  val_dataset.get_dataset_details()

  # ---- uncomment to plot class distribution ----
  # building_dataset.plot_class_distribution("../figures/class_distribution_validation.png")

  # data loaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  # experimentally cut the train_loader to 50 samples

  # Init baseline model 
  num_classes = len(train_dataset.classes)
  model = BaselineCNNCLassifier(num_classes).to(device)

  # loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training
  
  train_model(model,train_loader, criterion, optimizer, device, num_epochs=EPOCHS)
  # save checkpoint
  checkpoint = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'epoch': EPOCHS
  }
  torch.save(checkpoint, f'../checkpoints/baseline_cnn_{EPOCHS}.pth')
  print("Checkpoint saved")

  # Validation
  accuracy,all_preds, all_labels =  evaluate_model(model, val_loader, device, train_dataset.classes, confusion_matrix_path="../figures/baseline/confusion_matrix.png")

  
if __name__ == "__main__":
  main()
