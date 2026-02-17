# mlops-cats-dogs

mlops-cats-dogs







\#  MLOps Cats vs Dogs Classification Pipeline



End-to-end MLOps project implementing data preprocessing, model training, experiment tracking, CI/CD, containerization, deployment, monitoring, and automated smoke testing.



---



\##  Project Overview



This project demonstrates a complete MLOps lifecycle:



\- Dataset preprocessing and splitting

\- Baseline CNN model training

\- MLflow experiment tracking

\- FastAPI inference service

\- Docker containerization

\- Docker Compose deployment (CD)

\- Monitoring \& metrics

\- GitHub Actions CI/CD with smoke tests



---



\##  Project Structure



mlops-cats-dogs/

│

├── src/

│ ├── preprocess.py

│ ├── train.py

│ └── api.py

│

├── tests/

│ ├── test\_api.py

│ └── test\_preprocess.py

│

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── .gitignore

└── README.md







---



\##  Dataset



Source: Kaggle Cats vs Dogs Dataset



Data preprocessing includes:



\- Extraction

\- Corrupted image handling

\- Train/validation/test split

\- Image resizing and normalization



---



\## Model Training



\- Baseline CNN implemented using TensorFlow/Keras

\- Metrics logged with MLflow:

&nbsp; - Accuracy

&nbsp; - Loss

&nbsp; - Validation performance

\- Model artifacts generated locally (not tracked in Git)



---



\##  Experiment Tracking



MLflow used for:



\- Parameter tracking

\- Metric logging

\- Model artifact storage



Folder excluded from Git for clean repository management.



---



\## Inference Service (FastAPI)



Endpoints:



| Endpoint | Description |

|---------|------------|

| `/health` | Service health check |

| `/metrics` | Request count \& latency |

| `/predict` | Image classification |



Model loading is conditional to support CI/CD smoke testing without heavy artifacts.



---



\##  Dockerization



\### Build image:



```bash

docker build -t cats-dogs-api .



Deployment (Docker Compose)



docker compose up --build





Service exposed on:



http://localhost:8001





\## Monitoring



Built-in metrics:



Total requests served



Average latency



Available at: /metrics



\## CI/CD Pipeline (GitHub Actions)



Pipeline stages:



Install dependencies



Run unit tests



Build Docker image



Start service



\## Smoke tests:



Health check



Metrics check



\## Model artifacts excluded for lightweight CI builds (best MLOps practice).



Model Artifact Handling





Large model files are intentionally excluded from GitHub:



Avoids repository bloat



Avoids GitHub size limits



Follows real-world MLOps standards



In production, models would be loaded from:



MLflow registry



Cloud object storage (S3, Azure Blob, GCS)





\## Key MLOps Concepts Implemented



Data versioning ready

Reproducible training

Experiment tracking

Containerized deployment

CI/CD automation

Monitoring \& logging

Smoke testing

Artifact best practices







\## How to Run Locally





python src/preprocess.py

python src/train.py

docker compose up --build

