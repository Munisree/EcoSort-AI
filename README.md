# EcoSort AI

### Intelligent Waste Segregation and Sustainable Disposal Assistant

EcoSort AI is an AI-assisted sustainability prototype that helps users identify everyday waste from images and understand appropriate disposal, safety, recycling, and reuse practices.

The project is aligned with **UN Sustainable Development Goal 12: Responsible Consumption and Production**, with secondary alignment to **SDG 11: Sustainable Cities and Communities**.

---

## Problem

Incorrect waste segregation can cause recyclable materials to be mixed with general waste and can result in hazardous items such as batteries being disposed of incorrectly.

EcoSort AI addresses this challenge by combining image classification with a structured sustainability knowledge base to provide simple, category-specific waste management guidance.

---

## Solution

The system follows this workflow:

```text
User uploads waste image
        ↓
Image preprocessing
        ↓
EfficientNet-B0 classifier
        ↓
Predicted waste category + confidence
        ↓
Local knowledge retrieval
        ↓
Disposal + safety + recycling/reuse guidance
