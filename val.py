import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ESD-YOLO weights.")
    parser.add_argument("--weights", required=True, help="Path to the trained weights file.")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "val"), help="Output project directory.")
    parser.add_argument("--name", default="ESD-YOLO", help="Validation run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
