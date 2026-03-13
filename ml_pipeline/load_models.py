import os
import torch
import joblib
import numpy as np
import pandas as pd
import torchvision.models as models
import torch.nn as nn

device = torch.device("cpu")

# ---------------- PROJECT ROOT ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "mode_pkl")


# ================= CNN MODEL =================

cnn_model = models.efficientnet_b1()

cnn_model.classifier[1] = nn.Linear(
    cnn_model.classifier[1].in_features,
    4
)

cnn_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "cnn_breast_model.pth"),
        map_location=device
    )
)

cnn_model.eval()


# ================= RADIOMICS MODEL =================

radiomics_model = joblib.load(
    os.path.join(MODEL_DIR, "radiomics_model.pkl")
)


# ================= SCALER =================

radiomics_scaler = joblib.load(
    os.path.join(MODEL_DIR, "radiomics_globalscaler.pkl")
)


# ================= FEATURE FILTERS =================

variance_filter = joblib.load(
    os.path.join(MODEL_DIR, "variance_filter.pkl")
)

corr_features_to_drop = joblib.load(
    os.path.join(MODEL_DIR, "corr_features_to_drop.pkl")
)


# ================= FUSION MODEL =================

fusion_model = joblib.load(
    os.path.join(MODEL_DIR, "fusion_model.pkl")
)