# Active Learning Training Components - Part 2

## Enhanced Data Augmentation and Loading

```python
# Enhanced data augmentation focusing on medical images
train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # Reduced rotation for medical images
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.RandomResizedCrop(config.img_size, scale=(0.85, 1.0)),
    # Medical-specific augmentations
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
    # Add noise for robustness
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
])

val_test_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

## Training and Evaluation Functions

```python
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Enhanced training epoch with better monitoring"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = torch.zeros(config.num_classes)
    class_total = torch.zeros(config.num_classes)
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(config.num_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                class_total[i] += class_mask.sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Per-class accuracies
    class_accuracies = []
    for i in range(config.num_classes):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return epoch_loss, epoch_acc, class_accuracies

def evaluate_model(model, dataloader, criterion, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(dataloader.dataset)
    accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Detailed metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Tuberculosis-specific metrics
    tb_metrics = report.get('tuberculosis', {'precision': 0, 'recall': 0, 'f1-score': 0})
    
    metrics = {
        'loss': total_loss,
        'accuracy': accuracy,
        'tb_precision': tb_metrics['precision'] * 100,
        'tb_recall': tb_metrics['recall'] * 100,
        'tb_f1': tb_metrics['f1-score'] * 100,
        'report': report,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics
```

## Main Training Loop with Active Learning

```python
def active_learning_training_loop(al_framework, val_loader, test_loader, class_names):
    """Main active learning training loop"""
    
    # Initialize focal loss with class weights
    alpha_tensor = torch.tensor(config.focal_alpha).to(config.device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=config.focal_gamma)
    
    best_tb_f1 = 0.0
    best_model_path = 'best_active_learning_model.pth'
    
    for iteration in range(config.al_iterations):
        print(f"\n{'='*80}")
        print(f"ACTIVE LEARNING ITERATION {iteration + 1}/{config.al_iterations}")
        print(f"{'='*80}")
        
        # Create model for this iteration
        model = EnhancedChestXRayCNN(
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        ).to(config.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            verbose=True
        )
        
        scaler = torch.cuda.amp.GradScaler()
        
        # Get current labeled data
        train_loader = al_framework.get_labeled_dataloader(transform=train_transform)
        
        print(f"Training with {len(train_loader.dataset)} labeled samples")
        
        # Training history for this iteration
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
            
            # Train
            train_loss, train_acc, train_class_acc = train_epoch(
                model, train_loader, criterion, optimizer, scaler, config.device
            )
            
            # Validate
            val_metrics = evaluate_model(model, val_loader, criterion, config.device, class_names)
            
            # Update scheduler
            scheduler.step(val_metrics['loss'])
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"TB Precision: {val_metrics['tb_precision']:.2f}% | TB Recall: {val_metrics['tb_recall']:.2f}% | TB F1: {val_metrics['tb_f1']:.2f}%")
            
            # Early stopping based on TB F1 score
            if val_metrics['tb_f1'] > best_val_f1:
                best_val_f1 = val_metrics['tb_f1']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'iteration': iteration,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config.__dict__
                }, f'iteration_{iteration}_best_model.pth')
                
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model from this iteration
        checkpoint = torch.load(f'iteration_{iteration}_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        test_metrics = evaluate_model(model, test_loader, criterion, config.device, class_names)
        
        # Update performance tracking
        al_framework.update_performance(iteration, val_metrics, test_metrics)
        
        print(f"\nIteration {iteration + 1} Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"TB Precision: {test_metrics['tb_precision']:.2f}%")
        print(f"TB Recall: {test_metrics['tb_recall']:.2f}%")
        print(f"TB F1-Score: {test_metrics['tb_f1']:.2f}%")
        
        # Save best overall model
        if test_metrics['tb_f1'] > best_tb_f1:
            best_tb_f1 = test_metrics['tb_f1']
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'test_metrics': test_metrics,
                'config': config.__dict__,
                'class_names': class_names
            }, best_model_path)
            print(f"New best model saved! TB F1: {best_tb_f1:.2f}%")
        
        # Query new samples for next iteration (except last iteration)
        if iteration < config.al_iterations - 1:
            query_size = int(len(al_framework.unlabeled_indices) * config.query_size)
            query_size = min(query_size, len(al_framework.unlabeled_indices))
            
            if query_size > 0:
                queried_indices = al_framework.query_samples(model, query_size)
                print(f"Queried {len(queried_indices)} new samples for next iteration")
            else:
                print("No more samples to query - stopping early")
                break
        
        # Clear GPU memory
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    
    return al_framework.performance_history
```

This implements the core active learning framework. Would you like me to create the final parts including data loading, execution, and visualization components?