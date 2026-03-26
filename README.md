\# Credit Risk ML Platform

https://www.kaggle.com/datasets/beatafaron/loan-credit-risk-and-population-stability

Production-grade machine learning platform for predicting loan default risk using the Lending Club dataset

## DVC + S3 Setup

This project uses DVC with AWS S3 for model and data versioning. To run the backend and fetch models/data automatically, you must provide AWS credentials with access to the S3 bucket.

### 1. Request S3 Access

Ask the project owner for S3 bucket access (read-only or write, as needed). You will receive an AWS Access Key ID and Secret Access Key.

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```
cp .env.example .env
# Edit .env and add your AWS keys
```

### 3. Run Docker Compose

Build and start all services (backend will automatically fetch models/data from S3):

```
docker-compose up --build
```

**Note:** Never commit your AWS credentials to the repository.

---

This project demonstrates real-world ML engineering practices including:

\- hybrid ML pipelines

\- explainable AI

\- MLOps workflows

\- CI/CD pipelines

\- production deployment

The repository will evolve incrementally to reflect realistic engineering development.
