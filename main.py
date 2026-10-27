from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw

from config import HF_API_KEY

MODEL = "facebook/detr-resnet-101"
THRESHOLD = 0.5


def main():
    path = Path(input("Image path: ").strip().strip('"'))
    if not path.is_file():
        raise FileNotFoundError(path)

    image = Image.open(path).convert("RGB")
    detections = InferenceClient(model=MODEL, token=HF_API_KEY).object_detection(
        path, threshold=THRESHOLD
    )

    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection.box
        draw.rectangle(
            (box.xmin, box.ymin, box.xmax, box.ymax), outline="red", width=2
        )
        draw.text(
            (box.xmin, box.ymin),
            f"{detection.label} {detection.score:.2f}",
            fill="red",
        )

    output = path.with_name(f"annotated_{path.stem}.png")
    image.save(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
