## Deep Impact Ranking System

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

## Dashboard Preview

![Dashboard Preview](images/dashboard.png)