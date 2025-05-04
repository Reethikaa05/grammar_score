Grammar Scoring Engine
Build a Grammar Scoring Engine for Voice Samples (hosted on Kaggle).

Overview
The objective of this competition is to develop a Grammar Scoring Engine for spoken data samples. You are provided with an audio dataset where each file is between 45 to 60 seconds long. The ground truth labels are MOS Likert Grammar Scores for each audio instance (see rubric below). Your task is to build a model that takes an audio file as input and outputs a continuous score ranging from 0 to 5.

Your submission will be assessed based on your ability to preprocess the audio data, select an appropriate methodology to solve the problem, and evaluate its performance using relevant metrics.

Training: The training dataset consists of 444 samples.

Testing (Evaluation): The testing dataset consists of 195 samples.

Step 1: Project Architecture
grammar-scoring-engine/
├── notebooks/
│ └── Grammar_Scoring_Engine.ipynb
├── app/
│ ├── app.py
│ └── utils.py
├── models/
├── data/
│ ├── audios_train/
│ ├── audios_test/
│ ├── train.csv
│ └── test.csv
├── requirements.txt
└── README.md
