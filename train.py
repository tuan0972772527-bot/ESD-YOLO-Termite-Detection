import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    default_weights = WORKSPACE_ROOT / "yolo11n.pt"

    parser = argparse.ArgumentParser(description="Train ESD-YOLO on a termite dataset.")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "ESD-YOLO.yaml"),
        help="Path to the ESD-YOLO model YAML.",
    )
    parser.add_argument(
        "--weights",
        default=str(default_weights if default_weights.exists() else "yolo11n.pt"),
        help="Optional pretrained weights.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"), help="Output project directory.")
    parser.add_argument("--name", default="ESD-YOLO", help="Experiment name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=True,
        amp=False,
        close_mosaic=10,
        optimizer="SGD",
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
