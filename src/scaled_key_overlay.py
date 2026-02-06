import json
from pathlib import Path
import cv2

FULL_IMAGE = Path("outputs/key_steps/01_full_input.png")
WHITE_KEYS = Path("outputs/keys/white_keys.json")
BLACK_KEYS = Path("outputs/keys/black_keys.json")
OUT_IMAGE  = Path("outputs/keys/verify_keys_full.png")

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    img = cv2.imread(str(FULL_IMAGE))
    if img is None:
        raise FileNotFoundError(FULL_IMAGE)

    white = load_json(WHITE_KEYS)
    black = load_json(BLACK_KEYS)

    vis = img.copy()

    for v in white.values():
        cv2.rectangle(vis, (v["x_ul"], v["y_ul"]), (v["x_dr"], v["y_dr"]), (0,255,0), 2)

    for v in black.values():
        cv2.rectangle(vis, (v["x_ul"], v["y_ul"]), (v["x_dr"], v["y_dr"]), (0,0,255), 2)

    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMAGE), vis)
    print("Saved:", OUT_IMAGE)

if __name__ == "__main__":
    main()
