"""
Dataset classes for grocery shelf segmentation training.

This module provides PyTorch Dataset classes for loading and processing
grocery shelf images and segmentation masks.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GroceryShelfDataset(Dataset):
    """
    Dataset class for grocery shelf segmentation.
    
    Loads images and corresponding segmentation masks for training
    semantic segmentation models on grocery shelf products.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 10,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the processed data directory
            split: Data split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
            num_classes: Number of segmentation classes
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Load split manifest
        manifest_path = self.data_dir / f"{split}_split.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
            
        with open(manifest_path) as f:
            self.samples = json.load(f)
            
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing 'image' and 'mask' tensors
        """
        sample = self.samples[idx]
        
        # Load image and mask
        image_path = self.data_dir / sample["image_path"]
        mask_path = self.data_dir / sample["mask_path"]
        
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Default resize and convert to tensor
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and validate an image file."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image: {path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and validate a segmentation mask."""
        if not path.exists():
            raise FileNotFoundError(f"Mask not found: {path}")
            
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
            
        # Ensure mask values are in valid range
        mask = np.clip(mask, 0, self.num_classes - 1)
        return mask


class VideoFrameDataset(Dataset):
    """
    Dataset for loading sequential frames from videos.
    
    Useful for temporal consistency analysis and video processing.
    """
    
    def __init__(
        self,
        video_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        sequence_length: int = 5,
        image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initialize the video frame dataset.
        
        Args:
            video_dir: Directory containing video frames
            transform: Transform pipeline to apply
            sequence_length: Number of consecutive frames to load
            image_size: Target image size
        """
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.image_size = image_size
        
        # Group frames by video
        self.video_sequences = self._build_sequences()
        
    def _build_sequences(self) -> List[List[Path]]:
        """Build sequences of consecutive frames from videos."""
        sequences = []
        
        # Group frames by video prefix
        video_groups = {}
        for frame_path in self.video_dir.glob("*.jpg"):
            # Extract video identifier from filename
            video_id = "_".join(frame_path.stem.split("_")[:-1])
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(frame_path)
        
        # Create sequences within each video
        for video_id, frames in video_groups.items():
            frames.sort()  # Ensure chronological order
            
            for i in range(len(frames) - self.sequence_length + 1):
                sequence = frames[i:i + self.sequence_length]
                sequences.append(sequence)
                
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.video_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of frames."""
        sequence_paths = self.video_sequences[idx]
        frames = []
        
        for frame_path in sequence_paths:
            image = cv2.imread(str(frame_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = cv2.resize(image, self.image_size)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                
            frames.append(image)
        
        return {
            "frames": torch.stack(frames),
            "paths": [str(p) for p in sequence_paths],
        }


def get_transforms(
    image_size: Tuple[int, int] = (512, 512),
    split: str = "train",
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """
    Get albumentations transform pipeline for the specified split.
    
    Args:
        image_size: Target image size (height, width)
        split: Data split ('train', 'val', 'test')
        augmentation_config: Augmentation configuration
        
    Returns:
        Albumentations compose transform
    """
    if augmentation_config is None:
        augmentation_config = {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.1,
            "rotation": 15,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }
    
    if split == "train":
        transform_list = [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=augmentation_config["horizontal_flip"]),
            A.VerticalFlip(p=augmentation_config["vertical_flip"]),
            A.Rotate(
                limit=augmentation_config["rotation"],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=augmentation_config["brightness"],
                contrast=augmentation_config["contrast"],
                saturation=augmentation_config["saturation"],
                hue=augmentation_config["hue"],
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    else:
        # Validation and test - only resize and normalize
        transform_list = [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    
    return A.Compose(transform_list)


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    augmentation_config: Optional[Dict] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = GroceryShelfDataset(
        data_dir=data_dir,
        split="train",
        transform=get_transforms(image_size, "train", augmentation_config),
        image_size=image_size,
    )
    
    val_dataset = GroceryShelfDataset(
        data_dir=data_dir,
        split="val",
        transform=get_transforms(image_size, "val"),
        image_size=image_size,
    )
    
    test_dataset = GroceryShelfDataset(
        data_dir=data_dir,
        split="test",
        transform=get_transforms(image_size, "test"),
        image_size=image_size,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    from pathlib import Path
    
    # This would run if data is available
    data_dir = Path("../data/processed")
    if data_dir.exists():
        try:
            dataset = GroceryShelfDataset(data_dir, split="train")
            print(f"Dataset loaded successfully with {len(dataset)} samples")
            
            # Test loading a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {sample.keys()}")
                print(f"Image shape: {sample['image'].shape}")
                print(f"Mask shape: {sample['mask'].shape}")
        except Exception as e:
            print(f"Dataset test failed: {e}")
    else:
        print("No processed data found. Run preprocessing notebooks first.") 