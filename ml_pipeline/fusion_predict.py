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
    print("CNN probabilities:", cnn_probs)

    # -------- Radiomics Prediction --------
    radiomics_probs = radiomics_predict(image_path)
    print("Radiomics probabilities:", radiomics_probs)

    # -------- Combine Inputs --------
    fusion_input = np.hstack([cnn_probs, radiomics_probs]).reshape(1, -1)
    print("Fusion input vector:", fusion_input)

    # -------- Fusion Model Prediction --------
    final_probs = fusion_model.predict_proba(fusion_input)[0]
    print("Fusion probabilities:", final_probs)

    pred_class = int(np.argmax(final_probs))
    confidence = float(final_probs[pred_class])

    label = stage_map[pred_class]

    print("Predicted class:", pred_class)
    print("Predicted label:", label)
    print("Confidence:", confidence)
  

    return label, confidence