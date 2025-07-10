# 🗣️ Grammar Scoring Engine

A machine learning system designed to **automatically score spoken English grammar** from audio clips, predicting MOS Likert Grammar Scores (0–5) based on fluency, structure, and correctness. Built as part of a Kaggle-hosted challenge focused on **speech evaluation and modeling**.

## 🎯 Objective

Given 45–60 second voice samples, the goal is to:
- Preprocess spoken audio data
- Extract meaningful acoustic and linguistic features
- Train a regression model to predict grammar scores (0 to 5 scale)
- Evaluate the model on unseen audio clips using regression metrics (e.g., MAE, RMSE)

---

## 📁 Project Structure

grammar-scoring-engine/
├── notebooks/
│ └── Grammar_Scoring_Engine.ipynb # End-to-end analysis and model pipeline
├── app/
│ ├── app.py # CLI / API for model inference
│ └── utils.py # Preprocessing, feature extraction
├── models/ # Trained models, checkpoints
├── data/
│ ├── audios_train/ # Training audio samples
│ ├── audios_test/ # Test audio samples
│ ├── train.csv # Ground truth scores
│ └── test.csv # Test metadata
├── requirements.txt
└── README.md


---

## 🔍 Approach Summary

### 🧪 Preprocessing
- Standardized sampling rate
- Noise reduction, silence trimming
- Mel spectrogram & MFCC extraction

### 🧠 Feature Engineering
- **Acoustic features**: MFCCs, pitch, energy, pauses
- **Linguistic features** (optional): ASR transcript scoring, grammar complexity (via spaCy/NLP)

### 📈 Modeling Techniques
- **Baseline**: Ridge/Lasso Regression with MFCC
- **Advanced**: CNN + GRU over spectrograms (torch), ensemble of XGBoost + LSTM
- **Loss**: MSE/MAE with stratified KFold validation

### 🧪 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## ⚙️ How to Use

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Reethikaa05/grammar-scoring-engine.git
cd grammar-scoring-engine

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run notebook or inference
Use the notebook for training:

notebooks/Grammar_Scoring_Engine.ipynb
For prediction from audio:

python app/app.py --file data/audios_test/sample.wav

🧠 Sample Results
| Model             | MAE  | RMSE | R² Score |
| ----------------- | ---- | ---- | -------- |
| MFCC + Ridge      | 0.52 | 0.74 | 0.87     |
| CNN + GRU (Torch) | 0.41 | 0.61 | 0.91     |
| Ensemble (Best)   | 0.39 | 0.58 | 0.93     |


🚀 Highlights
📊 End-to-end voice scoring system

🎧 Real-time inference using PyDub and Librosa

🧠 Deep learning + classical ML ensemble

💬 Potential to integrate with speaking exam platforms or language apps

👩‍💻 Author
Reethika Selvam
GitHub: Reethikaa05
MCA | ML & AI Enthusiast | Speech + Vision Learner

📢 Looking to collaborate on AI for language learning, speech scoring, or real-time audio ML? Let’s connect!
