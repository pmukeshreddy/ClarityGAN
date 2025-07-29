import io, time, torch, os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torchvision.transforms.functional as TF
from model import MultiStageDeblurNet

app = FastAPI(title="GoPro Deblur Demo")

# ── serve static files under /static ──────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# root URL just returns index.html
@app.get("/")
def root():
    return FileResponse("static/index.html", media_type="text/html")

# ── load model once ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deblur_model_ema.pth"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  Model file {MODEL_PATH} not found!")
    MODEL = None
else:
    try:
        MODEL = MultiStageDeblurNet().to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        MODEL.eval()
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        MODEL = None

# ── POST /deblur ──────────────────────────────────────────────────────────
@app.post("/deblur")
async def deblur_image(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded properly")
        
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "PNG or JPG only")
    
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Bad image file")

    t0 = time.time()
    tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        sharp = MODEL(tensor)
    
    latency = time.time() - t0

    # Ensure output is properly clamped and converted
    out = TF.to_pil_image(sharp.squeeze().cpu().clamp(0, 1))
    
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"X-Processing-Time": f"{latency:.3f}"}
    )
