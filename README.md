# neu-steel-defect-EGT307

[Overview + objectives]
-----------------------

[Data Source section]
---------------------
We use the NEU Metal Surface Defects image dataset, organised into train, valid, and test directories. Each directory contains six defect categories: Crazing, Inclusion, Patches, Pitted, Rolled, and Scratches. The dataset is balanced across classes, with 276 training images per class (total 1,656 training images). The validation and test sets each contain 12 images per class (total 72 images for validation and 72 images for test). Overall, the dataset contains 1,800 images and is suitable for training and evaluating a 6-class steel surface defect classifier.

[Microservices setup]
---------------------
[How to run (Docker + K8s)]
---------------------------
## How to Run (Inference Service)

This project provides a FastAPI inference service that predicts NEU steel defect classes from an uploaded image.

### 1) Run locally (Python / venv)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r services\inference-service\requirements.txt

cd services\inference-service
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
Open:

http://127.0.0.1:8000/docs

http://127.0.0.1:8000/health

Note: http://127.0.0.1:8000/ may show 404 because no / route is defined. Use /docs or /health.

run docker with:
docker build -t neu-inference:1.0 services/inference-service
docker run --rm -p 8000:8000 neu-inference:1.0

test with:
curl.exe http://127.0.0.1:8000/health

Open:

http://127.0.0.1:8000/docs

Do NOT browse to http://0.0.0.0:8000. Use 127.0.0.1 or localhost.



[Advanced K8s features]
service and deployment yaml. to port forward: kubectl port-forward svc/neu-inference-svc 8000:8000
next : curl.exe http://127.0.0.1:30080/health
to test via node port : curl.exe http://127.0.0.1:30080/health


[Advanced K8s feature (HPA/Ingress)]
none so far

[Limitations]
this is only the foundation of the project more K8s features will be added, now it can only work small scale and only has deployment and service yaml.
