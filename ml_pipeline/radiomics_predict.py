import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import cv2
from radiomics import featureextractor

from ml_pipeline.load_models import (
    radiomics_model,
    radiomics_scaler,
    variance_filter,
    corr_features_to_drop
)

# ---------------- PROJECT ROOT ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "mode_pkl")

# ---------------- RADIOMICS EXTRACTOR ----------------

PARAM_PATH = os.path.join(MODEL_DIR, "global_strong_params.yaml")

extractor = featureextractor.RadiomicsFeatureExtractor(PARAM_PATH)

# ---------------- FEATURE ORDER ----------------

FEATURE_PATH = os.path.join(MODEL_DIR, "radiomics_featureGlobal_order.txt")

with open(FEATURE_PATH) as f:
    FEATURE_ORDER = [line.strip() for line in f]


# ---------------- FEATURE EXTRACTION ----------------

def get_radiomics_features(image_path):

    ext = os.path.splitext(image_path)[1].lower()

    # -------- Load Image --------
    if ext in [".dcm", ".dicom"]:
        dicom = pydicom.dcmread(image_path, force=True)
        img_array = dicom.pixel_array.astype(np.float32)

    elif ext in [".png", ".jpg", ".jpeg"]:
        img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    else:
        raise ValueError("Unsupported image format")

    # -------- Normalize --------
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

    # -------- Convert to SimpleITK --------
    image = sitk.GetImageFromArray(img_array)

    # -------- Create Mask --------

    img_uint8 = (img_array * 255).astype(np.uint8)

    _, mask_array = cv2.threshold(
        img_uint8,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask_array = mask_array.astype(np.uint8)

    # fallback if mask fails
    if mask_array.sum() < 50:
        mask_array = np.ones_like(img_array, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)

    # -------- Extract Radiomics --------
    result = extractor.execute(image, mask)

    features = []

    for name in FEATURE_ORDER:

        key = name.replace("global_", "")

        if key in result:
            features.append(float(result[key]))
        else:
            features.append(0.0)

    features = np.array(features, dtype=np.float32)

    return features


# ---------------- RADIOMICS PREDICTION ----------------

def radiomics_predict(image_path):

    # Step 1 — Extract features
    features = get_radiomics_features(image_path)

    # Step 2 — Convert to DataFrame
    features_df = pd.DataFrame([features], columns=FEATURE_ORDER)

    # Step 3 — Variance filter
    features = variance_filter.transform(features_df)
    features_df = pd.DataFrame(features)

    # Step 4 — Remove correlated features
    features_df = features_df.drop(columns=corr_features_to_drop, errors="ignore")

    features = features_df.values

    # Step 5 — Scale features
    features = radiomics_scaler.transform(features)

    # Step 6 — Predict
    probs = radiomics_model.predict_proba(features)[0]

    return probs