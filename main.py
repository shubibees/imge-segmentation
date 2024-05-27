from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import io
import torch
import uuid

app = FastAPI()

# Enable CORS for development (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for images
images = {}


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Convert the uploaded image to JPEG format
    image = Image.open(file.file).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Load the model and process the image
    processor = MaskFormerImageProcessor.from_pretrained(
        "facebook/maskformer-swin-large-ade")
    inputs = processor(images=image, return_tensors="pt")
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-large-ade")
    outputs = model(**inputs)

    # Post-process the semantic segmentation
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    predicted_semantic_map_np = predicted_semantic_map.cpu().numpy()

    # Store mask images in memory and generate URLs
    urls = []
    for class_label in range(predicted_semantic_map_np.max() + 1):
        mask = (predicted_semantic_map_np ==
                class_label).astype(np.uint8) * 255
        if np.any(mask):
            inverted_mask = 255 - mask
            inverted_mask_image = Image.fromarray(inverted_mask)
            inverted_mask_image = inverted_mask_image.convert("RGBA")
            datas = inverted_mask_image.getdata()
            newData = []
            for item in datas:
                if item[0] != 0 or item[1] != 0 or item[2] != 0:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            inverted_mask_image.putdata(newData)

            # Save the inverted mask image to memory
            mask_byte_arr = io.BytesIO()
            inverted_mask_image.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)
            unique_id = str(uuid.uuid4())
            images[unique_id] = mask_byte_arr

            # Generate URL for the mask
            urls.append(f"/mask/{unique_id}")

    return {"urls": urls}


@app.get("/mask/{unique_id}")
async def get_mask(unique_id: str):
    mask_byte_arr = images.get(unique_id)
    if mask_byte_arr:
        return StreamingResponse(mask_byte_arr, media_type="image/png")
    else:
        return {"error": "Mask not found"}, 404

