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
    ("phone",   re.compile(r'\b[6-9]\d{9}\b')),
    ("aadhaar", re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')),
    ("pan",     re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')),
    ("vehicle", re.compile(r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b')),
    ("email",   re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b')),
]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 10


def is_sensitive(text: str) -> bool:
    clean = text.replace(" ", "")
    return any(p.search(clean) or p.search(text) for _, p in PATTERNS)


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

    blurred_count = 0
    for (bbox, text, prob) in results:
        if prob < 0.3:
            continue
        if is_sensitive(text.replace(" ", "")):
            image = blur_region(image, bbox)
            blurred_count += 1

    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image")

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers={"X-Blurred-Regions": str(blurred_count)},
    )
