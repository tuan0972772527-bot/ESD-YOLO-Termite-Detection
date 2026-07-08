import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESD-YOLO inference.")
    parser.add_argument("--weights", required=True, help="Path to the trained weights file.")
    parser.add_argument("--source", required=True, help="Path to an image, folder, video, or stream.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument(
        "--project",
        default=str(PROJECT_ROOT / "runs" / "predict"),
        help="Output project directory.",
    )
    parser.add_argument("--name", default="ESD-YOLO", help="Prediction run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
