import numpy as np

from ml_pipeline.cnn_predict import get_cnn_features
from ml_pipeline.radiomics_predict import get_radiomics_features
from ml_pipeline.load_models import xgb_model, radiomics_scaler, hybrid_scaler




def hybrid_predict(image_path):

    cnn_features = get_cnn_features(image_path)

    print("CNN feature sample:", cnn_features[:5])

    radiomics_features = get_radiomics_features(image_path)

    print("Radiomics first 10:", radiomics_features[:10])
    print("Radiomics global part:", radiomics_features[33:40])

    radiomics_features = radiomics_scaler.transform([radiomics_features])[0]

    features = np.concatenate([cnn_features, radiomics_features])

    print("Hybrid vector size:", len(features))

    features = hybrid_scaler.transform([features])

    pred = xgb_model.predict(features)
    prob = xgb_model.predict_proba(features)

    stage_map = {
        0: "BI-RADS 1-2 (Normal/Benign)",
        1: "Stage 3 (Suspicious)",
        2: "Stage 4 (Malignant)",
        3: "Stage 5 (Highly Malignant)"
    }
    
    pred = xgb_model.predict(features)
    prob = xgb_model.predict_proba(features)[0]

    # override rule
    if prob[0] > 0.90:
        label = "BI-RADS 1-2 (Normal/Benign)"
    else:
        label = stage_map[int(np.argmax(prob))]

    print("Prediction probabilities:", prob)

    # -------- Confidence Scaling --------
    confidence = float(prob.max())

    confidence = 0.68 + (confidence * 0.17)
    confidence = min(confidence, 0.85)

    return label, confidence
