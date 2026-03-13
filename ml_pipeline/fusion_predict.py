import numpy as np

from ml_pipeline.cnn_predict import cnn_predict
from ml_pipeline.radiomics_predict import radiomics_predict
from ml_pipeline.load_models import fusion_model


stage_map = {
    0: "BI-RADS 1-2 (Normal/Benign)",
    1: "BI-RADS 3 (Suspicious)",
    2: "BI-RADS 4 (Malignant)",
    3: "BI-RADS 5 (Highly Malignant)"
}


def fusion_predict(image_path):

    # -------- CNN Prediction --------
    cnn_probs = cnn_predict(image_path)

    # -------- Radiomics Prediction --------
    radiomics_probs = radiomics_predict(image_path)

    # -------- Combine for Fusion Model --------
    fusion_input = np.hstack([cnn_probs, radiomics_probs]).reshape(1, -1)

    # -------- Final Prediction from Fusion Model --------
    final_probs = fusion_model.predict_proba(fusion_input)[0]

    pred_class = int(np.argmax(final_probs))
    confidence = float(final_probs[pred_class])

    label = stage_map[pred_class]

    return label, confidence