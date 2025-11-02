# 🧠 Customer Churn Prediction Service

## Problem

Subscription-based businesses lose significant revenue when customers stop using or paying for their service — a phenomenon known as **customer churn**.  
In many cases, companies only realize a customer has churned after it’s too late to intervene.

Predicting churn **before** it happens allows teams to take proactive actions such as:
- Offering personalized discounts or outreach.
- Improving onboarding or user experience.
- Strengthening customer relationships.

---

## Proposed Solution

We aim to build an **automated churn prediction system** that:
1. **Analyzes customer behavior** (usage, payments, support activity, etc.).
2. **Predicts churn probability** for each customer over the next 30 days.
3. **Exposes the model through a REST API** built with **FastAPI**.
4. **Runs in a Docker container** for portability and consistency.
5. **Deploys to AWS (ECS Fargate)** for scalability, with an eventual migration to **Kubernetes (EKS)** for orchestration and resilience.

---

## System Overview

### Architecture Overview
1. **Data Layer**  
   - Uses the IBM Telco Customer Churn dataset (public data simulating subscription customers).  
   - Preprocessing and feature engineering handled in Python scripts.

2. **Model Layer**  
   - Machine learning model (e.g., Gradient Boosting or Logistic Regression).  
   - Trained to classify customers as *likely to churn* or *likely to stay*.

3. **API Layer**  
   - FastAPI service provides `/predict` and `/health` endpoints.  
   - Accepts customer data, returns churn probability and (later) key drivers.

4. **Infrastructure Layer**  
   - Containerized using Docker.  
   - Deployed to AWS ECS Fargate, with CI/CD pipeline via GitHub Actions.  
   - Future migration to Kubernetes (EKS) with autoscaling and observability.

---

## Project Structure
churn-prediction-service/
├─ app/
│ ├─ init.py
│ ├─ main.py # FastAPI entry point
│ ├─ schemas.py # Request/response models
│ ├─ model.py # Model loading and prediction
├─ pipeline/
│ ├─ train.py # Training script
│ ├─ features.py # Feature engineering
├─ tests/
│ └─ test_app.py # Basic unit tests
├─ data/ # Raw data (gitignored)
├─ artifacts/ # Model outputs (gitignored)
├─ notebooks/ # Jupyter notebooks
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ Makefile
└─ README.md