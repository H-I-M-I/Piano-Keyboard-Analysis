import os
import json
import cv2

def main():
    os.makedirs("outputs", exist_ok=True)

    img = cv2.imread("outputs/frame_without_hands2.png")
    if img is None:
        raise ValueError("outputs/frame_without_hands2.png not found")

    # select ROI manually
    x, y, w, h = cv2.selectROI("Select Piano ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    roi = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    with open("outputs/piano_roi.json", "w") as f:
        json.dump(roi, f, indent=2)

    crop = img[y:y+h, x:x+w]
    cv2.imwrite("outputs/piano_roi_crop.png", crop)

    print("Saved: outputs/piano_roi.json")
    print("Saved: outputs/piano_roi_crop.png")

if __name__ == "__main__":
    main()
