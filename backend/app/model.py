import torch
from torch import nn
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDER_NAME = "all-mpnet-base-v2"
embedder = SentenceTransformer(EMBEDDER_NAME)


class ImpactRegressor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


model = ImpactRegressor().to(DEVICE)


def load_model(path="/media/u1970167/data/REF_impactor/models/deep_impact_model.pt"):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()


def embed_text(text: str):
    return embedder.encode([text], convert_to_tensor=True).to(DEVICE)


def embed_sentences(sentences):
    return embedder.encode(sentences, convert_to_tensor=True).to(DEVICE)


def predict_embedding(embedding):
    with torch.no_grad():
        return model(embedding).item()