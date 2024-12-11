Loan Default Prediction

Project Overview This project focuses on building a classification model to predict loan default behavior for a Non-Banking Financial Company (NBFC). The objective is to accurately identify potential loan defaulters and non-defaulters using historical loan data, enabling improved loan approval decisions and risk management strategies. The dataset comprises loan application details over the past two years, with the target variable being the loan status (1: default, 0: non-default).

Problem Statement The goal is to predict the loan default status based on customer demographic details, financial history, and loan attributes.

Dataset The dataset was provided in two parts:

train_data.xlsx: Used for training the model.
test_data.xlsx: Used for model evaluation.
Key Features:

customer_id: Unique ID for each customer.
**transaction_date: Date of the transaction.
sub_grade: Customer's sub-grade (based on factors like geography, income, and age).
term: Loan tenure.
home_ownership: Ownership status of the applicant's home.
cibil_score: Credit score of the applicant.
annual_inc: Annual income of the applicant.
int_rate: Interest rate on the loan.
purpose: Loan purpose.
loan_amnt: Loan amount.
emp_length: Employment experience in years.
loan_status: Target variable (1: default, 0: non-default).
Approach

Exploratory Data Analysis (EDA)

Analyzed feature distributions and relationships with the target variable.
Visualized data using histograms, boxplots, and correlation heatmaps to identify trends and outliers.
Data Preprocessing

Handled missing values and converted categorical variables using One-Hot Encoding.
Scaled numerical features to improve model performance.
Modeling

Built Logistic Regression and Random Forest models for classification.
Used a class-based approach to implement a modular pipeline with functions for data loading, preprocessing, training, testing, and predictions.
Hyperparameter Tuning

Tuned hyperparameters for both models to optimize performance.
Results

Logistic Regression Model:

Accuracy: 68.85%
Classification Report:
Precision (Defaulters): 70%
Recall (Defaulters): 86%
F1-Score (Defaulters): 77%
Key Insights: Logistic Regression provided good interpretability but struggled with complex relationships in the data.
Random Forest Model:

Accuracy: 68.57%
Classification Report:
Precision (Defaulters): 69%
Recall (Defaulters): 88%
F1-Score (Defaulters): 78%
Key Insights: Random Forest demonstrated a stronger ability to handle non-linear relationships and provided insights into feature importance.
Model Selection

Based on the evaluation metrics, Random Forest was selected as the final model for this project.

Reasons for Choosing Random Forest:

Better Recall for Defaulters: Random Forest achieved higher recall (88%) compared to Logistic Regression (86%), making it more effective at identifying defaulters.
Non-linear Relationships: It handles complex relationships between features that may exist in the dataset.
Feature Importance: Provides insights into the most critical features influencing predictions.
Balanced Performance: While both models had similar accuracy, Random Forest performed better in terms of recall and F1-score.
Conclusion The Random Forest model was chosen for its superior ability to classify defaulters accurately and its robustness with complex datasets. It is now ready for deployment to assist the NBFC in improving loan approval decisions and minimizing risk.

Files Included

eda.ipynb** - Exploratory Data Analysis.
model_pipeline.py** - Class-based model implementation (Logistic Regression and Random Forest).
model_selection.ipynb - Model selection and evaluation metrics.
random_forest_model.pkl - Trained Random Forest model.
