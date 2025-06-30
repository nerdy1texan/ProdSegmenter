"""
Unit tests for the training pipeline.

Tests cover dataset loading, model initialization, training loops,
and evaluation metrics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.dataset import GroceryShelfDataset, create_data_loaders, get_transforms


class TestDataset:
    """Test dataset functionality."""
    
    def test_grocery_shelf_dataset_initialization(self):
        """Test dataset initialization with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock manifest
            manifest_data = [
                {
                    "image_path": "frames/image_001.jpg",
                    "mask_path": "masks/mask_001.png"
                },
                {
                    "image_path": "frames/image_002.jpg", 
                    "mask_path": "masks/mask_002.png"
                }
            ]
            
            manifest_path = temp_path / "train_split.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
            
            # Test dataset creation (will fail due to missing images, but should initialize)
            with pytest.raises(FileNotFoundError):
                dataset = GroceryShelfDataset(temp_path, split="train")
    
    def test_get_transforms(self):
        """Test data augmentation transforms."""
        # Test training transforms
        train_transform = get_transforms(image_size=(512, 512), split="train")
        assert train_transform is not None
        
        # Test validation transforms
        val_transform = get_transforms(image_size=(512, 512), split="val")
        assert val_transform is not None
        
        # Test that training has more transforms than validation
        train_ops = len(train_transform.transforms)
        val_ops = len(val_transform.transforms)
        assert train_ops > val_ops


class TestModelComponents:
    """Test model components and utilities."""
    
    def test_dummy_model_creation(self):
        """Test creation of a dummy segmentation model."""
        # Simple segmentation model for testing
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=1)  # 10 classes
        )
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        output = model(input_tensor)
        
        expected_shape = (batch_size, 10, 256, 256)
        assert output.shape == expected_shape
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 10, kernel_size=1)
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable by default


class TestLossFunctions:
    """Test loss functions and metrics."""
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss computation."""
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy predictions and targets
        batch_size, num_classes, height, width = 2, 10, 64, 64
        predictions = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        loss = criterion(predictions, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_dice_loss_computation(self):
        """Test dice coefficient calculation."""
        def dice_coefficient(pred, target, num_classes):
            """Simple dice coefficient implementation."""
            dice_scores = []
            
            for cls in range(num_classes):
                pred_cls = (pred == cls).float()
                target_cls = (target == cls).float()
                
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                
                if union > 0:
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice.item())
                else:
                    dice_scores.append(1.0)  # Perfect score for empty class
            
            return np.mean(dice_scores)
        
        # Test with perfect prediction
        perfect_pred = torch.tensor([[0, 1, 2], [1, 2, 0]])
        perfect_target = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        dice_score = dice_coefficient(perfect_pred, perfect_target, num_classes=3)
        assert abs(dice_score - 1.0) < 1e-6  # Should be perfect


class TestTrainingLoop:
    """Test training loop components."""
    
    def test_training_step_simulation(self):
        """Simulate a training step."""
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 10, kernel_size=1)
        )
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)
        masks = torch.randint(0, 10, (batch_size, 128, 128))
        
        # Simulate training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_validation_step_simulation(self):
        """Simulate a validation step."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(16, 10, kernel_size=1)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)
        masks = torch.randint(0, 10, (batch_size, 128, 128))
        
        # Simulate validation step
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, masks)
            predictions = torch.argmax(outputs, dim=1)
        
        assert loss.item() > 0
        assert predictions.shape == masks.shape
        assert not torch.isnan(loss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_pixel_accuracy_calculation(self):
        """Test pixel accuracy computation."""
        # Perfect predictions
        predictions = torch.tensor([[0, 1, 2], [1, 2, 0]])
        targets = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        correct_pixels = (predictions == targets).sum().item()
        total_pixels = targets.numel()
        pixel_accuracy = correct_pixels / total_pixels
        
        assert pixel_accuracy == 1.0
        
        # Imperfect predictions
        predictions = torch.tensor([[0, 1, 2], [1, 1, 0]])  # One wrong pixel
        targets = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        correct_pixels = (predictions == targets).sum().item()
        total_pixels = targets.numel()
        pixel_accuracy = correct_pixels / total_pixels
        
        expected_accuracy = 5.0 / 6.0  # 5 correct out of 6 pixels
        assert abs(pixel_accuracy - expected_accuracy) < 1e-6
    
    def test_iou_calculation(self):
        """Test IoU (Intersection over Union) calculation."""
        def calculate_iou(pred, target, class_id):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            if union == 0:
                return 1.0  # Perfect IoU for empty class
            return intersection / union
        
        # Perfect prediction for class 0
        predictions = torch.tensor([[0, 0, 1], [0, 1, 1]])
        targets = torch.tensor([[0, 0, 1], [0, 1, 1]])
        
        iou_class_0 = calculate_iou(predictions, targets, class_id=0)
        iou_class_1 = calculate_iou(predictions, targets, class_id=1)
        
        assert iou_class_0 == 1.0
        assert iou_class_1 == 1.0


class TestModelSaving:
    """Test model checkpointing and saving."""
    
    def test_checkpoint_creation(self):
        """Test creation of model checkpoint."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 10, kernel_size=1)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5,
            'metrics': {'iou': 0.85, 'accuracy': 0.92}
        }
        
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 10
        assert checkpoint['metrics']['iou'] == 0.85
    
    def test_model_loading_simulation(self):
        """Test model loading from checkpoint."""
        # Create and save model
        original_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 10, kernel_size=1)
        )
        
        original_state = original_model.state_dict()
        
        # Create new model and load state
        new_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 10, kernel_size=1)
        )
        
        new_model.load_state_dict(original_state)
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            original_model.named_parameters(), 
            new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)


@pytest.fixture
def dummy_config():
    """Fixture providing dummy configuration."""
    return {
        'training': {
            'hyperparameters': {
                'batch_size': 4,
                'learning_rate': 0.001,
                'num_epochs': 1,
                'weight_decay': 0.0001
            },
            'optimizer': {
                'type': 'adamw',
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'loss': {
                'type': 'crossentropy'
            }
        },
        'data': {
            'preprocessing': {
                'resize_height': 256,
                'resize_width': 256
            }
        },
        'models': {
            'default': {
                'num_classes': 10,
                'architecture': 'dummy'
            }
        }
    }


class TestConfigurationHandling:
    """Test configuration loading and validation."""
    
    def test_config_validation(self, dummy_config):
        """Test configuration validation."""
        # Test required fields
        assert 'training' in dummy_config
        assert 'hyperparameters' in dummy_config['training']
        assert 'batch_size' in dummy_config['training']['hyperparameters']
        
        # Test value ranges
        assert dummy_config['training']['hyperparameters']['batch_size'] > 0
        assert dummy_config['training']['hyperparameters']['learning_rate'] > 0
        assert dummy_config['training']['hyperparameters']['num_epochs'] > 0
    
    def test_optimizer_config(self, dummy_config):
        """Test optimizer configuration."""
        optimizer_config = dummy_config['training']['optimizer']
        
        assert optimizer_config['type'] in ['adam', 'adamw', 'sgd']
        assert len(optimizer_config['betas']) == 2
        assert all(0 <= beta <= 1 for beta in optimizer_config['betas'])


class TestPerformanceBenchmarks:
    """Test performance requirements and benchmarks."""
    
    def test_inference_speed_simulation(self):
        """Test inference speed measurement."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, kernel_size=1)
        )
        
        model.eval()
        input_tensor = torch.randn(1, 3, 512, 512)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        
        # Measure inference time
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        fps = 1000.0 / avg_time
        
        assert avg_time > 0
        assert fps > 0
        
        # For this simple model, should be quite fast
        assert avg_time < 1000  # Less than 1 second per inference
    
    def test_memory_usage_estimation(self):
        """Test model memory usage estimation."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 10, kernel_size=1)
        )
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        assert model_size_mb > 0
        assert model_size_mb < 1000  # Should be reasonable for this simple model


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 