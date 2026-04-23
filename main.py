from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import easyocr
import cv2
import re
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'])

PATTERNS = [
    ("phone",   3, re.compile(r'\b[6-9]\d{9}\b')),
    ("aadhaar", 5, re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')),
    ("pan",     4, re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')),
    ("vehicle", 2, re.compile(r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b')),
    ("email",   2, re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b')),
    ("dob",     3, re.compile(
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'
        r'|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'
        r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'
        r')\b',
        re.IGNORECASE
    )),
]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 10


def match_sensitive(text: str):
    """Return (label, score) if text matches any pattern, else None."""
    clean = text.replace(" ", "")
    for label, score, pattern in PATTERNS:
        if pattern.search(clean) or pattern.search(text):
            return label, score
    return None


def blur_region(image: np.ndarray, bbox) -> np.ndarray:
    (tl, tr, br, bl) = bbox
    x1, y1 = max(0, int(tl[0])), max(0, int(tl[1]))
    x2, y2 = min(image.shape[1], int(br[0])), min(image.shape[0], int(br[1]))
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    k = max(51, (max(x2 - x1, y2 - y1) // 5) | 1)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return image


def compute_risk(detections: list[dict]) -> dict:
    """
    Score:   sum of per-detection weights, capped at 100
    Level:   LOW < 30  /  MEDIUM 30-59  /  HIGH 60-84  /  CRITICAL 85+
    """
    total = min(sum(d["score"] for d in detections), 100)

    if total >= 85:
        level, color = "CRITICAL", "#ff4d6d"
    elif total >= 60:
        level, color = "HIGH", "#f97316"
    elif total >= 30:
        level, color = "MEDIUM", "#facc15"
    else:
        level, color = "LOW", "#2ee8c0"

    return {"score": total, "level": level, "color": color}


@app.get("/")
def home():
    return {"message": "Privacy Detector running 🚀"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit")

    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    results = reader.readtext(image)

    detections = []
    for (bbox, text, prob) in results:
        if prob < 0.3:
            continue
        match = match_sensitive(text.replace(" ", ""))
        if match:
            label, score = match
            image = blur_region(image, bbox)
            detections.append({"type": label, "score": score, "text_len": len(text)})

    risk = compute_risk(detections)

    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image")

    # Encode detection types as comma-separated string for the header
    found_types = ",".join(sorted(set(d["type"] for d in detections))) or "none"

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers={
            "X-Blurred-Regions":  str(len(detections)),
            "X-Risk-Score":       str(risk["score"]),
            "X-Risk-Level":       risk["level"],
            "X-Risk-Color":       risk["color"],
            "X-Detected-Types":   found_types,
            "Access-Control-Expose-Headers": (
                "X-Blurred-Regions, X-Risk-Score, "
                "X-Risk-Level, X-Risk-Color, X-Detected-Types"
            ),
        },
    )
