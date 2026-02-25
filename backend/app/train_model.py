import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from utils import extract_text_from_pdf_path
from model import ImpactRegressor, DEVICE

EMBEDDER_NAME = "all-mpnet-base-v2"
MODEL_PATH = "/media/u1970167/data/REF_impactor/models/deep_impact_model.pt"

embedder = SentenceTransformer(EMBEDDER_NAME)


class PaperDataset(Dataset):
    def __init__(self, texts, scores):
        self.embeddings = embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        self.scores = torch.tensor(scores.values, dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.scores[idx]


def train(rp, pf, csvf):
    df = pd.read_csv(f"{rp}{csvf}")

    texts = []
    for path in tqdm(df["pdf_path"]):
        full_path = f"{rp}{pf}/{path}.pdf"
        texts.append(extract_text_from_pdf_path(full_path))

    dataset = PaperDataset(texts, df["score"])
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImpactRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(8):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")


if __name__ == "__main__":
    root_path = '/media/u1970167/data/REF_impactor/dataset/'
    paper_folder = 'papers'
    csv_file = 'training_data.csv'

    train(root_path, paper_folder, csv_file)