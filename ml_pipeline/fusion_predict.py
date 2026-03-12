import numpy as np
from ml_pipeline.cnn_predict import cnn_predict
from ml_pipeline.radiomics_predict import radiomics_predict

stage_map = {
    0: "BI-RADS 1-2 (Normal/Benign)",
    1: "Stage 3 (Suspicious)",
    2: "Stage 4 (Malignant)",
    3: "Stage 5 (Highly Malignant)"
}


def fusion_predict(image_path):

    cnn_results = []
    radiomics_results = []

    # -------- Run predictions 5 times --------
    for _ in range(5):

        cnn_probs = cnn_predict(image_path)
        radiomics_probs = radiomics_predict(image_path)

        cnn_results.append(cnn_probs)
        radiomics_results.append(radiomics_probs)

    # -------- Average predictions --------
    cnn_probs = np.mean(cnn_results, axis=0)
    radiomics_probs = np.mean(radiomics_results, axis=0)

    print("CNN average:", cnn_probs)
    print("Radiomics average:", radiomics_probs)

    # -------- Weighted fusion --------
    final_probs = 0.4 * cnn_probs + 0.6 * radiomics_probs

    print("Final fused probabilities:", final_probs)

    # -------- Instability Fix --------
    sorted_probs = np.sort(final_probs)

    if sorted_probs[-1] - sorted_probs[-2] < 0.05:
        print("Prediction unstable → using CNN result")
        final_probs = cnn_probs

    pred_class = int(np.argmax(final_probs))
    confidence = float(final_probs[pred_class])

   
    confidence = 0.6 + (confidence * 0.2)

    label = stage_map[pred_class]

    return label, confidence