import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import sys
import multiprocessing
import time


# Define dataset class outside main for Windows multiprocessing compatibility
class FastBirdSongDataset(Dataset):
    def __init__(self, dataframe, data_dir, img_col, label_col, class_mapping, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.img_col = img_col
        self.label_col = label_col
        self.class_mapping = class_mapping
        self.transform = transform
        
        # Pre-compute all file paths
        self.image_paths = [os.path.join(data_dir, row[img_col]) for _, row in dataframe.iterrows()]
        self.labels = [class_mapping[row[label_col]] for _, row in dataframe.iterrows()]
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Fast loading
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            # Return blank image if file can't be loaded
            print(f"Error loading {img_path}: {e}")
            blank_image = torch.zeros((3, 224, 224))
            return blank_image, label


# Model definition outside main for Windows multiprocessing compatibility
class BirdSongClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdSongClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adjust this based on your input size
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # Increased dropout from 0.5 to 0.6
            nn.Linear(128 * 14 * 14, 512),  # Adjusted for 224x224 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Increased dropout from 0.5 to 0.6
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Function to profile a dataloader iteration
def profile_dataloader(loader, device, num_batches=3):
    total_data_time = 0
    total_gpu_time = 0
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
            
        # Measure GPU transfer time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        total_gpu_time += elapsed_ms
        
        print(f"Batch {i}: Data shape: {images.shape}, GPU transfer: {elapsed_ms:.2f}ms")
    
    return total_data_time, total_gpu_time


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model, path='best_model.pth'):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# Main function
def main():
    # Force CUDA to use more memory and compute
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Force CUDA initialization with larger tensor to warm up
        dummy = torch.ones(2048, 2048, device=device)
        dummy = dummy @ dummy  # Matrix multiplication to engage CUDA cores
        del dummy  # Free memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("GPU initialized and ready")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")

    # Dataset parameters 
    EDDIE_SPECTROGRAM_DIR = 'spectrogram_eddie'
    EDDIE_CSV_FILE = 'labels_eddie.csv'
    EDDIE_CSV_LABEL = 'label'
    EDDIE_CSV_IMAGE = 'spectrogram_path'
    
    IMAGE_SIZE = (224, 224)  # Smaller for faster loading
    BATCH_SIZE = 412         
    EPOCHS = 50
    RANDOM_SEED = 112164
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait before early stopping

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Load data CSV
    print("Loading dataset info...")
    labels_df = pd.read_csv(EDDIE_CSV_FILE)
    print(f"CSV shape: {labels_df.shape}")

    # Get classes
    unique_classes = sorted(labels_df[EDDIE_CSV_LABEL].unique())
    num_classes = len(unique_classes)
    print(f"Classes: {num_classes}")

    # Create class mappings
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(unique_classes)}

    # Split dataset
    train_df, val_df = train_test_split(
        labels_df, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED, 
        stratify=labels_df[EDDIE_CSV_LABEL]
    )
    print(f"Training: {len(train_df)}, Validation: {len(val_df)}")

    # Fast data transformations (reduced operations)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Create datasets with optimized loading
    train_dataset = FastBirdSongDataset(
        train_df, 
        EDDIE_SPECTROGRAM_DIR, 
        EDDIE_CSV_IMAGE, 
        EDDIE_CSV_LABEL, 
        class_to_idx,
        transform=train_transform
    )

    val_dataset = FastBirdSongDataset(
        val_df, 
        EDDIE_SPECTROGRAM_DIR, 
        EDDIE_CSV_IMAGE, 
        EDDIE_CSV_LABEL, 
        class_to_idx,
        transform=val_transform
    )

    # Start with single loader, then try multiple if it works
    try:
        # Test with single process first
        print("Creating optimized data loaders...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=6,
            pin_memory=True,
            prefetch_factor=6,
            persistent_workers=True,  # Keep workers alive between batches
            drop_last=True  # Skip partial last batch for speed
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=6,
            pin_memory=True,
            prefetch_factor=6,
            persistent_workers=True,
            drop_last=True
        )
        
        # Run a quick profiling on the dataloaders
        print("Profiling dataloaders...")
        profile_dataloader(train_loader, device)
        
        print("Successfully created data loaders with worker processes")
    except Exception as e:
        print(f"Error with multiprocessing loaders: {e}")
        print("Falling back to single-process loading")
        
        # Fallback to single process
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

    print("Data loaders created")

    # Create model
    model = BirdSongClassifier(num_classes)
    model = model.to(device)
    print(model)

    # Set up optimizer with weight decay (L2 regularization) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)
    
    # Set up learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',          # Monitor validation loss
        factor=0.5,          # Reduce LR by half when triggered
        patience=5,          # Wait 5 epochs for improvement
        verbose=True,        # Print message when LR is reduced
        min_lr=1e-6          # Don't reduce LR below this value
    )
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Enable automatic mixed precision for faster training
    scaler = torch.amp.GradScaler('cuda')  # Updated syntax

    # Create CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Train the model with better profiling
    train_losses = []
    val_losses = []
    best_val_acc = 0.0
    batch_times = []
    
    # For plotting learning curves
    epochs_list = []
    train_acc_list = []
    val_acc_list = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        # Track timing for different operations
        data_load_times = []
        forward_times = []
        backward_times = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Record data load time
            data_load_time = time.time() - batch_start_time
            data_load_times.append(data_load_time)
            
            # Move data to GPU with timing
            torch.cuda.synchronize()
            start_event.record()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            end_event.record()
            torch.cuda.synchronize()
            gpu_transfer_time = start_event.elapsed_time(end_event) / 1000  # ms to seconds
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with timing
            start_event.record()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event) / 1000
            forward_times.append(forward_time)
            
            # Backward pass with timing
            start_event.record()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event) / 1000
            backward_times.append(backward_time)
            
            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate batch time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Show batch progress with detailed timing
            if batch_idx % 5 == 0:
                batch_acc = 100. * correct / total
                gpu_mem = torch.cuda.memory_allocated()/1e9
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, LR: {current_lr:.6f}, "
                      f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, "
                      f"Time: {batch_time:.3f}s (Data: {data_load_time:.3f}s, GPU→Transfer: {gpu_transfer_time:.3f}s, "
                      f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s), "
                      f"GPU Mem: {gpu_mem:.2f} GB")
                
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        
        # Calculate epoch time and component averages
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_data_time = sum(data_load_times) / len(data_load_times)
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_backward_time = sum(backward_times) / len(backward_times)
        batch_times = []  # Reset for next epoch
        data_load_times = []
        forward_times = []
        backward_times = []
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(avg_val_loss)
        val_time = time.time() - val_start_time
        
        # Store values for plotting
        epochs_list.append(epoch + 1)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print epoch results with detailed timing info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}")
        print(f"Time - Epoch: {epoch_time:.2f}s, Average Batch: {avg_batch_time:.3f}s")
        print(f"Breakdown - Data Loading: {avg_data_time:.3f}s, Forward: {avg_forward_time:.3f}s, "
              f"Backward: {avg_backward_time:.3f}s, Validation: {val_time:.2f}s")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_accuracy_model.pth")
            print(f"Saved best accuracy model with accuracy: {val_acc:.2f}%")
        
        # Check early stopping based on validation loss
        early_stopping(avg_val_loss, model, path='best_loss_model.pth')
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Force synchronize to clear any hanging CUDA operations
        torch.cuda.synchronize()
    
    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Models saved.")
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_acc_list, label='Training Accuracy')
    plt.plot(epochs_list, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to 'training_curves.png'")
    
    # Print performance summary
    print("\nTraining Performance Summary:")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training completed after {len(epochs_list)} epochs")
    
    # Show the gap between training and validation
    final_train_acc = train_acc_list[-1]
    final_val_acc = val_acc_list[-1]
    print(f"Final training accuracy: {final_train_acc:.2f}%")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Gap between training and validation: {final_train_acc - final_val_acc:.2f}%")


if __name__ == "__main__":
    # Important for Windows
    multiprocessing.freeze_support()
    main()