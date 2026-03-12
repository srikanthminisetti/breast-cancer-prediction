import os
import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from ml_pipeline.load_models import cnn_model


def cnn_predict(image_path):

    ext = os.path.splitext(image_path)[1].lower()

    # -------- DICOM --------
    if ext in [".dcm", ".dicom"]:
        dicom = pydicom.dcmread(image_path, force=True)
        image = dicom.pixel_array.astype(np.float32)

    # -------- PNG / JPG --------
    elif ext in [".png", ".jpg", ".jpeg"]:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    else:
        raise ValueError("Unsupported file format")

    # normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    image = torch.tensor(image).unsqueeze(0)

    # resize
    image = F.interpolate(
        image.unsqueeze(0),
        size=(224,224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # convert to 3 channels
    image = image.repeat(3,1,1)

    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(image)

        probs = torch.softmax(outputs, dim=1)

    return probs.numpy()[0]