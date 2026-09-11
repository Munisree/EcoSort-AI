# EcoSort AI — IBM Bob Development Log

This document records every significant interaction with **IBM Bob** during the
development of EcoSort AI. It serves as evidence of genuine AI-assisted
development for the 1M1B AI for Sustainability Virtual Internship
(IBM SkillsBuild × AICTE).

Each entry includes:
- The date and phase
- The question or task posed to Bob
- The outcome and decision made

---

## Entry 1 — Architecture and Planning

**Date:** Phase 0 (pre-coding)  
**Task posed to Bob:**  
Full project requirements provided. Bob was asked to analyse the requirements
and produce a complete implementation plan covering:

1. System architecture
2. Modular project/folder structure
3. Recommended AI components
4. How RAG should work
5. How the AI assistant/agent should work
6. Where IBM Bob fits into the development workflow
7. Whether and where IBM watsonx Orchestrate is useful
8. A realistic MVP
9. Optional advanced features
10. Responsible AI considerations
11. Testing and evaluation strategy
12. Technical risks and simpler alternatives

**Outcome:**  
Bob produced a comprehensive implementation plan with a Mermaid architecture
diagram, a 12-category technology table, a RAG sequence diagram, a tiered
watsonx assessment, an 8-week build order, a responsible AI table, a testing
strategy table, and a risk/alternative matrix.

**Key decisions made from this plan:**
- Use EfficientNet-B0 transfer learning on the Kaggle Garbage Classification
  dataset (12 classes)
- Use FAISS + sentence-transformers for RAG (local, zero-cost)
- Use local Flan-T5 or ollama as the LLM (watsonx.ai as upgrade path)
- Build incrementally: Phase 1 = classifier only
- IBM Bob = development partner, NOT the computer-vision model
- Maintain this log as documentary evidence of Bob's role

---

## Entry 2 — Phase 1 Scope Approval and First Revision

**Date:** Phase 1 planning  
**Task posed to Bob:**  
Architecture approved. Bob was asked to plan Phase 1 only:
- Dataset preparation and stratified train/val/test split
- EfficientNet-B0 transfer learning
- Evaluation metrics
- Saving the trained model
- Streamlit UI for image upload and prediction display

**Initial response reviewed and revised:**  
Bob initially proposed building Phase 1 with a randomly initialised
classification head, describing predictions as "illustrative."

**Why this was rejected:**  
A randomly initialised head produces predictions that are statistically
meaningless — equivalent to a biased random number generator. Presenting
random outputs as AI classification would be academically dishonest, would
undermine the integrity of the project, and would not constitute a working
prototype. Even in a demonstration context, displaying incorrect results
labelled as AI predictions could mislead users about proper waste disposal.
This was correctly identified and rejected.

**Revised plan approved with the following requirements:**
1. Phase 1 must include actual fine-tuning — no random/untrained predictions.
2. The model must be absent → hard refusal, not random output.
3. Class names must be read from the dataset folder structure (not hard-coded).
4. Stratified split (80/10/10) using per-class indices.
5. Class names and index mapping saved inside the checkpoint.
6. Evaluation: test accuracy + sklearn classification_report.
7. Use `EfficientNet_B0_Weights.DEFAULT` (current torchvision API).
8. Confidence shown as "model's estimated confidence", not a correctness guarantee.

---

## Entry 3 — Phase 1 Final Implementation

**Date:** Phase 1 implementation  
**Task posed to Bob:**  
Final corrections approved. Bob was asked to create all 10 Phase 1 files:

| File | Role |
|---|---|
| `requirements.txt` | Phase 1 dependencies |
| `classifier/__init__.py` | Package definition |
| `classifier/labels.py` | Runtime class-name reader from ImageFolder |
| `classifier/preprocess.py` | PIL → normalised tensor, training + val transforms |
| `classifier/model.py` | build, save, load, predict — EfficientNet-B0 |
| `classifier/train.py` | Stratified split, two-phase training, evaluation |
| `models/.gitkeep` | Directory placeholder |
| `app.py` | Streamlit UI — hard refusal if model absent |
| `docs/bob_log.md` | This file |
| `.gitignore` | Exclude model weights, dataset, pycache |

**Key implementation decisions made with Bob's assistance:**

- `save_checkpoint()` persists `class_names`, `class_to_idx`, and `num_classes`
  inside the `.pth` file so that inference never requires the dataset.
- `load_trained_model()` reads `num_classes` from the checkpoint — `app.py`
  never hard-codes 12.
- `stratified_split()` uses two-stage `StratifiedShuffleSplit` from scikit-learn
  to guarantee proportional class representation in train / val / test.
- Two-phase fine-tuning: backbone frozen for `WARMUP` epochs, then the last
  `UNFREEZE_N` MBConv blocks are unfrozen with a 10× lower learning rate.
- `app.py` calls `st.stop()` if the checkpoint file does not exist — no path
  to displaying uninitialised predictions.
- EXIF transpose applied to uploaded images to handle rotated phone photos.
- `ReduceLROnPlateau` scheduler on validation loss with patience=3.
- Best checkpoint saved on each validation loss improvement, not at final epoch.

**Training command:**
```bash
python -m classifier.train
```

**Next phase:** Phase 2 — RAG pipeline (knowledge-base ingestion, FAISS vector
store, LLM-based disposal recommendations). To be started after Phase 1
training is complete and validated.

---

*This log will be updated at the start of each new development phase.*
