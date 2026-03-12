import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from radiomics import featureextractor

from ml_pipeline.load_models import (
    radiomics_model,
    radiomics_scaler,
    variance_filter,
    corr_features_to_drop
)

# Radiomics extractor
extractor = featureextractor.RadiomicsFeatureExtractor(
    "mode_pkl/global_strong_params.yaml"
)

# Feature order used during training (58 features)
with open("mode_pkl/radiomics_featureGlobal_order.txt") as f:
    FEATURE_ORDER = [line.strip() for line in f]


# ---------------- FEATURE EXTRACTION ----------------

def get_radiomics_features(image_path):

    # Read DICOM
    import os
    import cv2

    ext = os.path.splitext(image_path)[1].lower()

    # -------- DICOM --------
    if ext in [".dcm", ".dicom"]:
        dicom = pydicom.dcmread(image_path, force=True)
        img_array = dicom.pixel_array.astype(np.float32)

    # -------- PNG / JPG --------
    elif ext in [".png", ".jpg", ".jpeg"]:
        img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    else:
        raise ValueError("Unsupported image format")

    # Normalize like CNN preprocessing
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

    # Convert to SimpleITK image
    image = sitk.GetImageFromArray(img_array)

    # Create mask (remove background)
    threshold = np.percentile(img_array, 30)

    mask_array = (img_array > threshold).astype(np.uint8)

    # Safety fallback if mask becomes empty
    if mask_array.sum() < 100:
        mask_array = np.ones_like(img_array, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)

    print("Mask pixels:", mask_array.sum())

    # Extract radiomics
    result = extractor.execute(image, mask)

    features = []

    # Maintain same feature order used during training
    for name in FEATURE_ORDER:

        # remove "global_" prefix because PyRadiomics does not include it
        key = name.replace("global_", "")

        if key in result:
            features.append(float(result[key]))
        else:
            features.append(0.0)

    features = np.array(features, dtype=np.float32)

    return features


# ---------------- PREDICTION ----------------

def radiomics_predict(image_path):

    # Step 1 — Extract 58 features
    features = get_radiomics_features(image_path)

    print("Radiomics raw features sample:", features[:10])

    # Step 2 — Convert to dataframe
    features_df = pd.DataFrame([features], columns=FEATURE_ORDER)

    # Step 3 — Variance filter
    features = variance_filter.transform(features_df)

    print("After variance filter:", features.shape)

    features_df = pd.DataFrame(features)

    # Step 4 — Drop correlated features
    features_df = features_df.drop(columns=corr_features_to_drop, errors="ignore")

    print("After correlation filter:", features_df.shape)

    features = features_df.values

    # Step 5 — Scale features
    features = radiomics_scaler.transform(features)

    print("Final radiomics vector:", features)

    # Step 6 — Predict
    probs = radiomics_model.predict_proba(features)[0]

    print("Radiomics probabilities:", probs)

    return probs
