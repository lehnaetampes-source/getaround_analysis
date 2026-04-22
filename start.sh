#!/bin/bash
# Lance FastAPI sur le port 7860 (port principal HF) et Streamlit sur 8501
uvicorn api:app --host 0.0.0.0 --port 7860 &
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
