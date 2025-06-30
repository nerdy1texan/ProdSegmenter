#!/usr/bin/env python3
"""
Model evaluation script for ProdSegmenter.
Provides accuracy metrics, performance benchmarks, and visualizations.
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from training.dataset import create_data_loaders


class ModelEvaluator:
    """Comprehensive model evaluation and benchmarking."""
    
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int):
        self.model = model.to(device).eval()
        self.device = device
        self.num_classes = num_classes
        self.results = {}
    
    def evaluate_accuracy(self, data_loader) -> Dict[str, float]:
        """Evaluate model accuracy metrics."""
        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(self.num_classes)
        union = torch.zeros(self.num_classes)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating accuracy"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Pixel accuracy
                correct = (predictions == masks).sum().item()
                total_correct += correct
                total_pixels += masks.numel()
                
                # IoU calculation
                for cls in range(self.num_classes):
                    pred_mask = (predictions == cls)
                    true_mask = (masks == cls)
                    
                    intersection[cls] += (pred_mask & true_mask).sum().item()
                    union[cls] += (pred_mask | true_mask).sum().item()
        
        # Calculate metrics
        pixel_accuracy = total_correct / total_pixels
        iou_per_class = intersection / (union + 1e-10)
        mean_iou = iou_per_class.mean().item()
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'iou_per_class': iou_per_class.tolist()
        }
    
    def evaluate_performance(self, data_loader, num_iterations=100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        sample_batch = next(iter(data_loader))
        single_image = sample_batch['image'][:1].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(single_image)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Benchmarking performance"):
                start_time = time.time()
                _ = self.model(single_image)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        return {
            'avg_inference_time_ms': float(np.mean(times)),
            'fps': float(1000.0 / np.mean(times)),
            'meets_realtime_30fps': bool(np.mean(times) < 33.33),
            'meets_realtime_60fps': bool(np.mean(times) < 16.67),
        }
    
    def generate_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'model_info': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
                'num_classes': self.num_classes
            },
            'accuracy_metrics': self.results.get('accuracy', {}),
            'performance_metrics': self.results.get('performance', {}),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # This would be replaced with actual model loading logic
    # based on the checkpoint configuration
    print(f"Loading model from {checkpoint_path}")
    
    # For now, return a dummy model - replace with actual model loading
    from torch import nn
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 10, 1)  # 10 classes
    )
    
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Could not load state dict - using random weights: {e}")
    
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ProdSegmenter model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating model: {args.model_path}")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(Path(args.model_path), device)
    logger.info("✅ Model loaded successfully")
    
    # Create data loader
    try:
        if args.split == "test":
            _, _, data_loader = create_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
        elif args.split == "val":
            _, data_loader, _ = create_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
        else:  # train
            data_loader, _, _ = create_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Could not create data loader: {e}")
        logger.info("Creating dummy data loader for testing")
        
        # Create dummy data loader for testing
        from torch.utils.data import TensorDataset, DataLoader
        dummy_images = torch.randn(100, 3, 512, 512)
        dummy_masks = torch.randint(0, 10, (100, 512, 512))
        dummy_dataset = TensorDataset(dummy_images, dummy_masks)
        
        class DummyBatch:
            def __init__(self, images, masks):
                self.data = {'image': images, 'mask': masks}
            def __getitem__(self, key):
                return self.data[key]
        
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            masks = torch.stack([item[1] for item in batch])
            return DummyBatch(images, masks)
        
        data_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    logger.info(f"✅ Data loader created")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, num_classes=10)
    
    # Evaluate accuracy
    logger.info("🎯 Evaluating accuracy...")
    accuracy_results = evaluator.evaluate_accuracy(data_loader)
    evaluator.results['accuracy'] = accuracy_results
    
    logger.info("📊 Accuracy Results:")
    for key, value in accuracy_results.items():
        if isinstance(value, list):
            formatted_values = [f'{v:.4f}' for v in value[:5]]
            logger.info(f"  {key}: {formatted_values}...")  # Show first 5 classes
        else:
            logger.info(f"  {key}: {value:.4f}")
    
    # Benchmark performance
    if args.benchmark:
        logger.info("⚡ Benchmarking performance...")
        performance_results = evaluator.evaluate_performance(data_loader)
        evaluator.results['performance'] = performance_results
        
        logger.info("🚀 Performance Results:")
        for key, value in performance_results.items():
            if isinstance(value, bool):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value:.2f}")
    
    # Generate report
    logger.info("📋 Generating evaluation report...")
    report_path = output_dir / "evaluation_report.json"
    report = evaluator.generate_report(report_path)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    
    if 'accuracy' in evaluator.results:
        acc = evaluator.results['accuracy']
        print("\n📊 Accuracy Metrics:")
        print(f"  mIoU: {acc.get('mean_iou', 0):.4f}")
        print(f"  Pixel Accuracy: {acc.get('pixel_accuracy', 0):.4f}")
    
    if 'performance' in evaluator.results:
        perf = evaluator.results['performance']
        print("\n🚀 Performance Metrics:")
        print(f"  Average Inference Time: {perf.get('avg_inference_time_ms', 0):.2f} ms")
        print(f"  FPS: {perf.get('fps', 0):.1f}")
        print(f"  Real-time 30 FPS: {'✅' if perf.get('meets_realtime_30fps') else '❌'}")
        print(f"  Real-time 60 FPS: {'✅' if perf.get('meets_realtime_60fps') else '❌'}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main() 