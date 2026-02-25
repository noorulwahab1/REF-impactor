from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.utils import extract_text_from_pdf_bytes, split_into_sentences
from app.model import load_model, embed_text, embed_sentences, predict_embedding

app = FastAPI(title="Impact Ranking Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_model()


@app.post("/rank")
async def rank(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        text = extract_text_from_pdf_bytes(contents)

        doc_embedding = embed_text(text)
        doc_score = predict_embedding(doc_embedding)

        sentences = split_into_sentences(text)
        sentence_embeddings = embed_sentences(sentences)

        sentence_scores = []
        for i, emb in enumerate(sentence_embeddings):
            score = predict_embedding(emb.unsqueeze(0))
            sentence_scores.append((sentences[i], score))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "filename": file.filename,
            "predicted_impact_score": round(doc_score, 2),
            "top_sentences": [
                {
                    "sentence": s[0],
                    "importance_score": round(s[1], 3)
                }
                for s in sentence_scores[:5]
            ]
        })

    results.sort(key=lambda x: x["predicted_impact_score"], reverse=True)

    return results