#!/usr/bin/env python3

import argparse
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed.")
    print("Please install: pip install ultralytics")
    sys.exit(1)


def main():
    """Train YOLOv8 model with specified dataset."""
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for insect detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset config file (data.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    args = parser.parse_args()

    # Start training
    print("Starting YOLOv8 training...")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")

    try:
        # Load pre-trained model and train
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=640
        )

        # Display final results
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Model saved: {results.save_dir}/weights/best.pt")

        # Display mAP metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
