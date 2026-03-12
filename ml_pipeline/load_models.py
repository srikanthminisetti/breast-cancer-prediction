import torch
import joblib
import numpy as np
import pandas as pd
import torchvision.models as models
import torch.nn as nn

device = torch.device("cpu")


# ================= CNN MODEL =================

cnn_model = models.efficientnet_b1()

cnn_model.classifier[1] = nn.Linear(
    cnn_model.classifier[1].in_features,
    4
)

cnn_model.load_state_dict(
    torch.load(
        "/Users/srikanthminisetti/Documents/Thesis_Final_work/NoteBooks/pkl_files/cnn_breast_model.pth",
        map_location=device
    )
)

cnn_model.eval()


# ================= RADIOMICS MODEL =================

radiomics_model = joblib.load(
    "/Users/srikanthminisetti/Documents/Thesis_Final_work/NoteBooks/pkl_files/radiomics_model.pkl"
)


# ================= SCALER =================

radiomics_scaler = joblib.load(
    "/Users/srikanthminisetti/Documents/Thesis_Final_work/notebook/radiomics_globalscaler.pkl"
)


# ================= FEATURE FILTERS =================

variance_filter = joblib.load(
    "/Users/srikanthminisetti/Documents/Thesis_Final_work/notebook/variance_filter.pkl"
)

corr_features_to_drop = joblib.load(
    "/Users/srikanthminisetti/Documents/Thesis_Final_work/notebook/corr_features_to_drop.pkl"
)