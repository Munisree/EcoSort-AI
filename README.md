````markdown
# EcoSort AI

### Intelligent Waste Segregation and Sustainable Disposal Assistant

EcoSort AI is an AI-assisted sustainability prototype that helps users identify waste from images and provides appropriate disposal, safety, recycling, and reuse guidance.

## SDG Alignment

- **SDG 12 – Responsible Consumption and Production** (Primary)
- **SDG 11 – Sustainable Cities and Communities** (Secondary)

## How It Works

```text
Waste Image
    ↓
Image Preprocessing
    ↓
EfficientNet-B0 Classification
    ↓
Waste Category + Confidence
    ↓
Local Knowledge Retrieval
    ↓
Disposal & Safety Guidance
````

## Key Features

* AI-based waste image classification
* 12 waste categories
* Confidence-based prediction
* Disposal and safety guidance
* Recycling and reuse suggestions
* Local knowledge retrieval
* Streamlit web interface
* Privacy-conscious image processing

## Technologies

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Transfer Learning
* Streamlit
* Scikit-learn
* Local Knowledge Retrieval
* IBM Bob

## Model Performance

* **Test Accuracy:** 95.55%
* **Macro F1 Score:** 0.939
* **Validation Accuracy:** 95.81%

## Dataset

Garbage Classification dataset with 12 waste categories.

The local dataset contains **15,515 images** and is used for training, validation, and testing.

## Project Structure

```text
EcoSort-AI/
├── app.py
├── classifier/
├── rag/
├── knowledge_base/
├── models/
├── docs/
├── requirements.txt
└── .gitignore
```

The dataset and trained model checkpoint are excluded from the repository to keep it lightweight.

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m streamlit run app.py
```

## Responsible AI

The prototype displays prediction confidence, provides low-confidence warnings, includes safety guidance for hazardous waste, and advises users to follow local municipal disposal instructions.

## Project Status

**Completed Prototype**

Developed as part of the **1M1B AI for Sustainability Virtual Internship**.

## Author

**Muni Sree Vundela**
B.Tech – Computer Science and Systems Engineering
Andhra University College of Engineering


