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
from transformers import pipeline
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    pipeline
)
## constants
TRAIN_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/train" # 5591 images
VAL_DIR = "/scratch/user/hasnat.md.abdullah/IDRT_classification_model/data/5 Classification example - TAMU image analytics/data/validation" # 859 images





def evaluate_model(model, processor, val_dataset, device, classes, confusion_matrix_path):
    # model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        pbar = tqdm(range(len(val_dataset)), desc='Evaluating')
        
        for idx in pbar:
            _, labels,img_path = val_dataset[idx]
            prompt = f"Question: What type of building is this? Options: {', '.join(classes)}. Instruction: Only answer within the options Answer:"
            print(f"Prompt: {prompt}")
            inputs = processor(Image.open(img_path), text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            predicted_label = processor.decode(outputs[0], skip_special_tokens=True).strip()
            # print(f"Predicted label: {predicted_label}")
            # print(f"types: {type(predicted_label)}")
            # print(predicted_label.split('Answer: ')[1].split()[0].lower())
            # print(predicted_label.split('Answer: ')[1].split()[1])
            # print(f"predicted label: {predicted_label}")
            if len(predicted_label.split('Answer: ')) > 1:
                print(f"predicted label: {predicted_label}")
                if len(predicted_label.split('Answer: ')[1].split()) > 1:
                    predicted_label = predicted_label.split('Answer: ')[1].split()[0].lower()+" "+predicted_label.split('Answer: ')[1].split()[1]
            else:
                predicted_label = ""
            try: 
                predicted_idx = val_dataset.class_to_idx[predicted_label]
                all_preds.append(predicted_idx)
                all_labels.append(labels)

                
                correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
                acc = 100 * correct / len(all_preds)
                pbar.set_postfix({'accuracy': f'{acc:.2f}%'})
            except KeyError:
                print(f"Predicted label: {predicted_label} not in classes")
                continue
    print(f"Total predictions: {len(all_preds)}")
    accuracy = 100.0 * sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
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
    
    # Model initialization
    checkpoint = "Salesforce/blip2-opt-2.7b"
    blip2_processor = Blip2Processor.from_pretrained(checkpoint, cache_dir="cache")
    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, cache_dir="cache").to(device)

    # Data transformations (validation)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets and loaders
    val_dataset = BuildingDataset(VAL_DIR, transform=transform_val)
    
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
          
    # Final evaluation
    accuracy, all_preds, all_labels = evaluate_model(
        model, blip2_processor, val_dataset, device, val_dataset.classes,
        confusion_matrix_path="../figures/blip2/blip2_confusion_matrix_class6.png" #TODO
    )

if __name__ == "__main__":
    main()
