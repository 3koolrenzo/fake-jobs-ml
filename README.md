## 🕵️‍♂️ Fake Job Posting Detector — Machine Learning Project

This project uses Natural Language Processing (NLP) and Machine Learning to detect fake job postings based on job description text.
The model is trained using the Kaggle Real or Fake Job Posting Prediction Dataset and classifies each listing as either:

Real Job (0)

Fake Job (1)

The model uses TF-IDF text vectorization + Logistic Regression, and includes a confusion matrix to visualize performance.

## 📚 Dataset

Source:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Dataset includes:

17,880 total job posts

17,014 real jobs

866 fake jobs

Key columns used:

text → Job description

fraudulent → 0 = real job, 1 = fake job

## 📊 Model Performance
Fake Job Class (1)

Precision: 1.00

Recall: 0.45

F1 Score: 0.62

Real Job Class (0)

Precision: 0.97

Recall: 1.00

F1 Score: 0.99

Overall Metrics

Accuracy: 0.97

Macro Avg F1: 0.80

This means:

The model almost never misclassifies real jobs as fake.

It detects around 45% of fake jobs, which is expected for imbalanced datasets.

This provides a strong baseline for a traditional ML approach.

## 🖼️ Confusion Matrix

(Saved at: reports/confusion_matrix.png)

## 🛠️ How to Run This Project (macOS)
1. Create & activate the virtual environment
python3 -m venv venv
source venv/bin/activate

2. Install the required libraries
pip install -r requirements.txt


If you don’t have a requirements file yet:

pip freeze > requirements.txt

3. Place the dataset correctly

Place the downloaded CSV at:

data/fake_job_postings.csv

4. Run the training script
python src/train_model.py


This will:

Train the machine learning model

Save the trained classifier and TF-IDF vectorizer to /models

Generate the confusion matrix in /reports
