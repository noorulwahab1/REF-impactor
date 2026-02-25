## Deep Impact Ranking System (Prototype)

A Transformer-based NLP system that predicts and ranks the academic impact of research papers directly from PDF files.

This project combines pretrained sentence embeddings with a neural regression model to estimate the potential influence of academic papers and identify the most impactful sentences within them.

## Overview

This system allows users to:

Upload multiple research papers (PDF format)

Predict an impact score for each paper

Rank papers by predicted impact

Highlight the most influential sentences within each document

The backend is built with FastAPI and PyTorch.
The frontend is built with React (Vite).

##How It Works
1) PDF Text Extraction

PDF files are processed using PyMuPDF (fitz), which extracts raw text from each page.

2) Semantic Embedding (Transformer)

Each document is converted into a dense semantic vector using:

all-mpnet-base-v2

Provided by SentenceTransformers

This model generates a 768-dimensional embedding that captures contextual meaning of the full document.

Instead of keyword matching, the system understands semantic structure.

3) Neural Impact Regressor

The embedding is passed into a fully connected neural network:

768 -> 512 -> 128 -> 1

Framework: PyTorch

Loss: Mean Squared Error

Optimizer: Adam

The model learns:

Document embedding -> Impact score

Sentence-Level Attribution

4) To improve interpretability:

The document is split into sentences

Each sentence is embedded individually

Each sentence is scored independently

Top 5 highest scoring sentences are returned

This provides a proxy explanation of what drives predicted impact.

System Architecture
User Upload (React) ->
FastAPI Backend ->
PDF Text Extraction (PyMuPDF) ->
Transformer Embedding (MPNet) ->
Neural Regression Model (PyTorch) ->
Sentence-Level Scoring ->
Ranking + JSON Response ->
Frontend Dashboard Display

## Running the Project Locally

Backend

Create environment:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run server:

uvicorn app.main:app --reload

Backend runs on:

http://localhost:8000
Frontend
cd frontend
npm install
npm run dev

Frontend runs on:

http://localhost:5173
Training the Model

To retrain:

python app/train_model.py

This will:

Load training dataset

Extract text from PDFs

Generate embeddings

Train neural regression model

Save deep_impact_model.pt

## Dashboard Preview

![Dashboard Preview](images/dashboard.png)