## How to Run the Project

Follow these steps to set up and run the Grammar Scoring Engine project:

1. **Clone the Repository**  
    Clone the project repository to your local machine:
    ```bash
    git clone <repository_url>
    cd grammar-scoring-engine
    ```

2. **Set Up a Virtual Environment**  
    Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**  
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the Data**  
    Ensure the `data/` directory contains the following:
    - `audios_train/` and `audios_test/` directories with audio files.
    - `train.csv` and `test.csv` files with metadata.

5. **Run the Notebook**  
    Open and execute the Jupyter notebook to preprocess data and train the model:
    ```bash
    jupyter notebook notebooks/Grammar_Scoring_Engine.ipynb
    ```

6. **Run the Application**  
    Start the application to test the Grammar Scoring Engine:
    ```bash
    python app/app.py
    ```

7. **Evaluate the Model**  
    Use the provided testing dataset to evaluate the model's performance.

8. **Modify and Extend**  
    Update the `utils.py` or `app.py` files to customize the application as needed.
 SHL-Hiring-Assessment_project
Build a Grammar Scoring Engine for Voice Samples (hosted on Kaggle).


## Overview

The objective of this competition is to develop a Grammar Scoring Engine for spoken data samples. You are provided with an audio dataset where each file is between 45 to 60 seconds long. The ground truth labels are MOS Likert Grammar Scores for each audio instance (see rubric below). Your task is to build a model that takes an audio file as input and outputs a continuous score ranging from 0 to 5.

Your submission will be assessed based on your ability to preprocess the audio data, select an appropriate methodology to solve the problem, and evaluate its performance using relevant metrics.

Training: The training dataset consists of 444 samples.

Testing (Evaluation): The testing dataset consists of 195 samples.

## Step 1: Project Architecture

grammar-scoring-engine/
├── notebooks/
│   └── Grammar_Scoring_Engine.ipynb
├── app/
│   ├── app.py
│   └── utils.py
├── models/
├── data/
│   ├── audios_train/
│   ├── audios_test/
│   ├── train.csv
│   └── test.csv
├── requirements.txt
└── README.md
