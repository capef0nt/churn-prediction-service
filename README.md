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
   - Machine learning model (Logistic Regression).  
   - Trained to classify customers as *likely to churn* or *likely to stay*.

3. **API Layer**  
   - FastAPI service provides `/predict` and `/health` endpoints.  
   - Accepts customer data, returns churn probability and (later) key drivers.

4. **Infrastructure Layer**  
   - Containerized using Docker.  
   - Deployed to AWS ECS Fargate, with CI/CD pipeline via GitHub Actions.  
   - Future migration to Kubernetes (EKS) with autoscaling and observability.

**Running**  
Architecture

The deployed architecture runs as a containerized web service on AWS ECS (Fargate).
It uses an Application Load Balancer (ALB) to route incoming HTTP requests to running containers.

Client → ALB (HTTP 80) → ECS Service → Fargate Task (Docker Container)

Local Development

Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate


Install dependencies

pip install -r requirements.txt


Run the FastAPI app

uvicorn app.main:app --reload


Access the API

Swagger docs: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

Containerization

Build the Docker image

docker build -t churn-api:latest .


Run the container locally

docker run --rm -p 8000:8000 --name churn-api churn-api:latest


Verify the app

curl http://127.0.0.1:8000/health


Expected output:

{"status": "ok", "model_loaded": true}

Deployment on AWS ECS Fargate
Overview

The trained model is packaged into a FastAPI microservice and deployed using Amazon ECS Fargate.
This approach enables a scalable, serverless deployment that automatically manages container orchestration, networking, and scaling.

Steps

1. Build and Tag the Docker Image

docker build -t churn-api:latest .


2. Create an ECR Repository

aws ecr create-repository --repository-name churn-api


3. Authenticate Docker with ECR

aws ecr get-login-password --region <your-region> \
| docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com


4. Push the Image to ECR

docker tag churn-api:latest <account-id>.dkr.ecr.<your-region>.amazonaws.com/churn-api:latest
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/churn-api:latest


5. Deploy to ECS Fargate

Create a Task Definition using the ECR image.

Configure:

CPU: 0.5 vCPU

Memory: 1 GB

Environment Variable: MODEL_PATH=/app/artifacts/model.joblib

Attach an Application Load Balancer (ALB) targeting port 8000.

Create a Service to run one or more tasks.

6. Access the Service
Once deployment is complete and the service is healthy:

http://<alb-dns-name>/health
http://<alb-dns-name>/docs

Monitoring and Scaling

Logs: View in Amazon CloudWatch Logs under /ecs/churn-api.

Autoscaling: ECS Fargate services can scale based on CPU or memory utilization.

Networking: Runs within a secure VPC with ALB handling inbound traffic.

--- ----- ---- ----- ----- ----- 


tech_stack:
  languages:
    - Python
  libraries:
    - pandas
    - numpy
    - scikit-learn
    - fastapi
  tools:
    - Docker
    - Uvicorn
    - Joblib
  cloud_services:
    - AWS ECR
    - AWS ECS Fargate
    - AWS CloudWatch
    - AWS Application Load Balancer

model:
  algorithm: Logistic Regression
  target_variable: churn_flag
  input_features:
    - customer demographics
    - subscription details
    - service usage metrics

deployment:
  approach: Containerized FastAPI application
  infrastructure:
    - AWS ECS (Fargate) cluster
    - Application Load Balancer for routing
    - CloudWatch Logs for monitoring
  steps:
    - Build Docker image locally
    - Push image to Amazon ECR
    - Create ECS Task Definition referencing image
    - Deploy ECS Service with Fargate launch type
    - Expose public API through ALB on port 8000

api_endpoints:
  - path: /health
    method: GET
    description: Health check endpoint returning service status
  - path: /predict
    method: POST
    description: Accepts customer data and returns churn probability
  - path: /docs
    method: GET
    description: Auto-generated API documentation via FastAPI Swagger UI

outputs:
  health_response: '{"status": "ok", "model_loaded": true}'
  prediction_example: '{"churn_probability": 0.78, "churn_flag": 1}'

author:
  name: B C. Marimbita
  contact: bcm637@gmail.com


