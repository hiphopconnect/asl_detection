import os
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

# Berechne den relativen Pfad zum VideoFiles-Ordner (Annahme: Dieser liegt im Projektstamm)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "VideoFiles")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    print("Received a POST request for upload")
    if file.content_type != "video/mp4":
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed.")
    file_contents = await file.read()
    if len(file_contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 500 MB.")
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_contents)
    return {"info": f"File '{file.filename}' was successfully uploaded.", "filename": file.filename}
