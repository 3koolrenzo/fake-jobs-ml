## Fake Job Posting Detector — Machine Learning Project
This project uses machine learning to detect fake job postings based on job description text. The model is trained using the Kaggle Real or Fake Job Posting Dataset and determines whether a job listing is:

Real Job (0)
Fake Job (1)

This project trains a text-classification model using TF-IDF + Logistic Regression and produces a confusion matrix image showing performance.

## Dataset
Source:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

The dataset includes:
17,880 total job posts
17,014 real jobs
866 fake jobs

Columns used:
text — job description
fraudulent — 0 = real job, 1 = fake job
The dataset is highly imbalanced, which impacts recall for fake jobs.

## Model Performance
Metric	Fake Job Class
Precision	1.00
Recall	0.45
F1 Score	0.62

## Interpretation
Precision = 1.0 → When the model predicts a fake job, it is always correct.
Recall = 0.45 → The model detects about 45% of all fake jobs (missing some due to imbalance).
Accuracy ≈ 97% → Strong performance driven by many real job examples.

## Confusion Matrix (Saved in: reports/confusion_matrix.png)
Fake Job Posting Detection Confusion Matrix
3403 real jobs correctly classified
0 real jobs incorrectly labeled fake
78 fake jobs correctly classified
95 fake jobs missed (predicted as real)

This shows the model is conservative:
✔ It never falsely calls a real job fake
✘ But it sometimes misses actual scams

## How to Run This Project (macOS)
Follow these steps exactly to reproduce the model.

1️. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install the required libraries
pip install -r requirements.txt

If you don’t have a requirements file yet:
pip freeze > requirements.txt

3️.Place the dataset correctly

Put the downloaded CSV at:

data/fake_job_postings.csv

4️. Run the training script
python src/train_model.py
