#!/usr/bin/env python3
"""
Main training script for ProdSegmenter models.

This script orchestrates the complete training process including:
- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Checkpoint saving and metric logging
- Azure ML integration
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.dataset import create_data_loaders
from training.utils.metrics import SegmentationMetrics
from training.utils.losses import get_loss_function
from training.utils.scheduler import get_scheduler

# Azure ML imports (optional)
try:
    from azureml.core import Run
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Initialize model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized PyTorch model
    """
    architecture = config['models']['default']['architecture']
    num_classes = config['models']['default']['num_classes']
    
    if architecture == 'deeplabv3plus':
        from training.models.deeplabv3plus import DeepLabV3Plus
        backbone = config['models']['deeplabv3plus']['backbone']
        output_stride = config['models']['deeplabv3plus']['output_stride']
        pretrained = config['models']['default']['pretrained']
        
        model = DeepLabV3Plus(
            num_classes=num_classes,
            backbone=backbone,
            output_stride=output_stride,
            pretrained=pretrained
        )
        
    elif architecture == 'unet':
        from training.models.unet import UNet
        encoder_name = config['models']['unet']['encoder_name']
        encoder_weights = config['models']['unet']['encoder_weights']
        decoder_channels = config['models']['unet']['decoder_channels']
        
        model = UNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            classes=num_classes
        )
        
    elif architecture == 'yolov8_seg':
        from training.models.yolov8_seg import YOLOv8Seg
        model_size = config['models']['yolov8_seg']['model_size']
        
        model = YOLOv8Seg(
            model_size=model_size,
            num_classes=num_classes
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    run = None
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        run: Azure ML run context (optional)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })
        
        # Log to Azure ML
        if run and AZURE_ML_AVAILABLE:
            run.log('batch_loss', loss.item())
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} - Average training loss: {avg_loss:.4f}")
    
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    metrics: SegmentationMetrics,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    run = None
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        metrics: Metrics calculator
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
        run: Azure ML run context (optional)
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    metrics.reset()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Update metrics
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions, masks)
    
    # Calculate final metrics
    avg_loss = total_loss / len(val_loader)
    metric_results = metrics.compute()
    
    results = {
        'val_loss': avg_loss,
        **metric_results
    }
    
    logger.info(f"Epoch {epoch} - Validation results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
        
        # Log to Azure ML
        if run and AZURE_ML_AVAILABLE:
            run.log(key, value)
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✅ New best model saved: {best_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ProdSegmenter model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="prodsegmenter_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['hyperparameters']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['hyperparameters']['batch_size'] = args.batch_size  
    if args.learning_rate:
        config['training']['hyperparameters']['learning_rate'] = args.learning_rate
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize Azure ML run (if available)
    run = None
    if AZURE_ML_AVAILABLE:
        try:
            run = Run.get_context()
            logger.info("Azure ML run context found")
        except:
            logger.info("No Azure ML context, running locally")
    
    # Initialize Weights & Biases (if requested)
    if args.wandb:
        wandb.init(
            project="prodsegmenter",
            config=config,
            name=f"train_{config['models']['default']['architecture']}"
        )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config['training']['hyperparameters']['batch_size'],
        image_size=(
            config['data']['preprocessing']['resize_height'],
            config['data']['preprocessing']['resize_width']
        ),
        augmentation_config=config['training']['augmentation']
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = get_model(config)
    model = model.to(device)
    
    # Initialize loss function
    criterion = get_loss_function(config['training']['loss'])
    
    # Initialize optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['hyperparameters']['learning_rate'],
            weight_decay=config['training']['hyperparameters']['weight_decay'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Initialize scheduler
    scheduler = get_scheduler(optimizer, config['training']['scheduler'])
    
    # Initialize metrics
    metrics = SegmentationMetrics(
        num_classes=config['models']['default']['num_classes'],
        device=device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['metrics'].get('mean_iou', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best mIoU: {best_metric:.4f}")
    
    # Training loop
    num_epochs = config['training']['hyperparameters']['num_epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, logger, run
        )
        
        # Validate
        val_results = validate_epoch(
            model, val_loader, criterion, metrics,
            device, epoch, logger, run
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step(val_results['val_loss'])
        
        # Check if best model
        current_metric = val_results.get('mean_iou', 0.0)
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_results, 
            config, checkpoint_dir, is_best
        )
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                **val_results,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Log to Azure ML
        if run and AZURE_ML_AVAILABLE:
            run.log('epoch', epoch)
            run.log('train_loss', train_loss)
            run.log('learning_rate', optimizer.param_groups[0]['lr'])
    
    logger.info("Training completed!")
    logger.info(f"Best validation mIoU: {best_metric:.4f}")
    
    # Register best model in Azure ML
    if run and AZURE_ML_AVAILABLE:
        best_model_path = checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            run.upload_file('outputs/best_model.pth', str(best_model_path))
            
            # Register the model
            run.register_model(
                model_name='prodsegmenter',
                model_path='outputs/best_model.pth',
                model_framework='PyTorch',
                description=f'Grocery shelf segmentation model - {config["models"]["default"]["architecture"]}'
            )
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 