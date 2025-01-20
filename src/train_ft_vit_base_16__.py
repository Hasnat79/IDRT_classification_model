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
from models.vit_finetune import ViTFineTune
from tqdm import tqdm
import random
## constants
TRAIN_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/train" # 5591 images
VAL_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/validation" # 859 images
EPOCHS = 20
WEIGHT_DECAY=0.01 # L2 regularization

class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        area = img.size()[1] * img.size()[2]
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, self.r2)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            if img.size()[0] == 3:
                img[0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                img[1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                img[2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
            else:
                img[0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
        return img
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}')
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'accuracy': f'{accuracy:.2f}%'})
    
    return losses.avg, accuracy

def validate(model, val_loader, criterion, device, epoch, num_epochs):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{num_epochs}')
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            accuracy = 100 * correct / total
            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'accuracy': f'{accuracy:.2f}%'})
    
    return losses.avg, accuracy

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=5, checkpoint_path="checkpoints/model.pth"):
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Train phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    # Plot training curves #TODO
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, "../figures/training_curves/class6_train_ft_vit_epoch20__new.png")
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, val_loader, device, classes, confusion_matrix_path):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
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
            
            acc = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{acc:.2f}%'})
    
    accuracy = 100.0 * correct / total
    print(f"Final Validation accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return accuracy, all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data transformations (vanilla) 
    # transform_train = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                        std=[0.229, 0.224, 0.225])
    # ])
    # stronger data augmentation
    transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # Added vertical flip
    transforms.RandomRotation(degrees=15),  # Increased rotation range
    transforms.RandomAffine(
        degrees=15, 
        translate=(0.15, 0.15),  # Increased translation
        scale=(0.8, 1.2),  # Added scale augmentation
        shear=10  # Added shear augmentation
    ),
    transforms.ColorJitter(
        brightness=0.3,  # Increased color jittering
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),  # Randomly convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    RandomErasing(probability=0.5)  # Added random erasing
])

    # Data transformations (validation)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets and loaders
    train_dataset = BuildingDataset(TRAIN_DIR, transform=transform_train)
    val_dataset = BuildingDataset(VAL_DIR, transform=transform_val)
    
    print("===== Train Dataset details =====")
    train_dataset.get_dataset_details()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model initialization
    num_classes = len(train_dataset.classes)
    model = ViTFineTune(num_classes=num_classes).to(device)


    # 
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': (p for n, p in model.named_parameters() if 'head' not in n),
            'lr': 1e-5,
            'weight_decay': WEIGHT_DECAY # L2 regularization
         
         },
        {'params': model.vit.head.parameters(),
          'lr': 1e-4,
           'weight_decay': WEIGHT_DECAY*5}, # Stronger L2 for the head
        
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=EPOCHS,
                                                      eta_min=1e-6 # minimum lr
                                                      )
    
    # Training
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=EPOCHS, checkpoint_path="../checkpoints/epoch20_vit_ft_class6_new.pth" #TODO
    )
    # --- uncomment to load checkpoint from path
    # checkpoint = torch.load("/scratch/user/hasnat.md.abdullah/IDRT_classification_model/checkpoints/epoch20_vit_ft_class9_new.pth")
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # Final evaluation
    accuracy, all_preds, all_labels = evaluate_model(
        model, val_loader, device, val_dataset.classes,
        confusion_matrix_path="../figures/vit_ft/epoch20_confusion_matrix_class6__new.png" #TODO
    )

if __name__ == "__main__":
    main()
