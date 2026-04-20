# Loan Eligibility Prediction

##1)Project Overview
In this project, I worked on predicting whether a loan application would be approved or rejected based on an applicant’s financial and personal details. The goal was to understand how factors like income, credit score, and assets influence loan approval decisions, and to build a model that can make these predictions accurately.

## Objective
The main objective of this project is to build a machine learning classification model that can determine loan eligibility. Along the way, the focus is also on understanding the data, identifying key patterns, and interpreting which factors play the most important role in the approval process.

## Tools & Technologies
- Python for data analysis and modeling  
- Pandas & NumPy for data handling  
- Matplotlib & Seaborn for visualization  
- Scikit-learn for machine learning models  
- Jupyter Notebook / Google Colab for development  
- GitHub for version control and project sharing  

## Dataset Description
The dataset contains detailed information about loan applicants, including their financial background and personal attributes. Key features include:

- Number of dependents  
- Education level  
- Employment status  
- Annual income  
- Loan amount and loan term  
- CIBIL (credit) score  
- Asset values (residential, commercial, luxury, and bank assets)  
These features help in understanding the applicant’s financial stability and creditworthiness.

### Target Variable:
- `loan_status` → Indicates whether the loan is approved or rejected  
This variable is what the model ultimately learns to predict.# HexSoftwares_Loan_Eligibility_Prediction
Loan Eligibility Prediction using Machine Learning (EDA + Model Building)

## 2)Data Understanding
I began by exploring the dataset to understand its structure, data types, and overall quality. The dataset contains 4,269 records and 13 columns, covering applicant profile details, income, loan characteristics, credit score, and asset values.
The data includes both numerical and categorical features. Variables such as income, loan amount, CIBIL score, and asset values are numerical, while education, self-employment status, and loan status are categorical.
A key observation at this stage was that the dataset is already well-structured, with no missing values across any column. This made the preprocessing step more straightforward and allowed the analysis to focus more on understanding feature behavior and model performance.
The `loan_id` column was identified as an identifier rather than a predictive feature, so it was excluded from modeling.

## 3)Data Cleaning
The dataset required only minimal cleaning before analysis. I first standardized the column names by removing extra spaces and replacing them with underscores, which made them easier to reference in Python.
Next, I removed the `loan_id` column because it serves only as a unique identifier and does not contribute to predicting loan eligibility.
I then checked for missing values and found that the dataset contains no null values. I also verified that there were no duplicate records. Since the data was already clean and complete, no imputation or duplicate removal was required.
Overall, this step confirmed that the dataset was ready for exploratory analysis and model building with only light preprocessing.

## 4)Exploratory Data Analysis (EDA)
I performed exploratory data analysis to better understand how different applicant and financial attributes relate to loan approval.
The loan status distribution provided an initial view of how approvals and rejections are represented in the dataset, which is important for understanding class balance before modeling.
The boxplot comparing CIBIL score and loan status revealed one of the clearest patterns in the project: applicants with approved loans tend to have noticeably higher CIBIL scores than rejected applicants. This suggests that credit history is a strong factor in loan approval decisions.
The income analysis showed that approved applicants generally appear to have stronger annual income profiles, indicating that repayment capacity likely influences approval outcomes.
The loan amount analysis suggested that loan size also affects approval behavior, although its relationship appears more nuanced than CIBIL score. This indicates that lenders may consider the requested amount along with other financial indicators rather than in isolation.
The residential assets analysis also pointed to a pattern where applicants with stronger asset holdings appear more likely to receive loan approval, reflecting the importance of overall financial stability.
Finally, the correlation heatmap helped identify relationships among numerical features such as income, loan amount, and asset values, providing useful intuition for model development.

## 5)Feature Engineering and Encoding
Before training the machine learning models, I converted the categorical variables into numerical form using Label Encoding.
The variables `education`, `self_employed`, and `loan_status` were encoded so that they could be processed by the classification algorithms. This step was necessary because most machine learning models in scikit-learn require numeric input.
I also explored feature scaling as part of preprocessing, but the final models in this project were trained on the encoded feature set without using the scaled version. Since tree-based models such as Decision Tree and Random Forest do not require feature scaling, this did not affect their performance.

## 6)Data Splitting
After preprocessing, the dataset was divided into training and testing sets.  
80% of the data was used to train the models, while 20% was reserved for testing.
This approach ensures that the final evaluation reflects how well the models perform on unseen data rather than only on the data used for training.

## 7)Model Building
To predict loan eligibility, I trained three classification models:
- Logistic Regression
- Decision Tree
- Random Forest
Logistic Regression was used as a baseline model, while Decision Tree and Random Forest were used to capture more complex and non-linear patterns in the data.
This comparison helped evaluate whether a simpler linear model was sufficient or whether ensemble learning could achieve better predictive performance.

## 8)Model Evaluation
The models were evaluated using accuracy, confusion matrix, precision, recall, and F1-score.
The Logistic Regression model achieved an accuracy of about 79.9%, which provided a useful baseline but was clearly weaker than the tree-based approaches.
Both Decision Tree and Random Forest achieved an accuracy of about 97.8%, showing a major improvement over Logistic Regression. Among these, Random Forest was selected as the final model because of its strong overall performance and more reliable generalization.
The Random Forest confusion matrix showed that the model correctly classified most loan applications, with only a small number of false positives and false negatives. It performed especially well on rejected loans, while still maintaining strong performance on approved loans.
The classification report confirmed this balanced performance, with precision and recall values close to 98% overall. Recall for class 0 was particularly strong at 0.99, while class 1 also showed solid recall at 0.96.

## 9)Results & Insights
The final Random Forest model achieved an accuracy of approximately 97.8%, making it the strongest model in this project.
Several important insights emerged from the analysis:
- **CIBIL score appears to be one of the strongest indicators of loan approval**, as approved applicants consistently show higher credit scores in the exploratory analysis.
- **Income is also an important factor**, with approved applicants generally showing stronger annual income levels.
- **Asset-related features contribute meaningful financial context**, suggesting that overall financial stability influences loan decisions.
- **Tree-based models performed far better than Logistic Regression**, indicating that the relationship between applicant characteristics and loan approval is not purely linear.
- **The final model was well balanced across both classes**, with high precision and recall for both approved and rejected applications.
Overall, the results suggest that loan eligibility is influenced by a combination of creditworthiness, repayment capacity, and asset strength rather than a single variable alone.

## 10)Conclusion
In this project, I developed a machine learning solution to predict loan eligibility using applicant financial and personal data.
The analysis began with data understanding and preprocessing, where I confirmed that the dataset was clean, complete, and suitable for modeling. Through exploratory analysis, I identified strong patterns linking CIBIL score, income, and asset values with loan approval outcomes.
I then trained and compared three machine learning models: Logistic Regression, Decision Tree, and Random Forest. The Random Forest model delivered the best overall performance, achieving approximately 97.8% accuracy with strong precision, recall, and F1-scores across both classes.
This project demonstrates how machine learning can support more efficient and data-driven loan approval decisions. It also shows that combining applicant credit behavior with financial strength indicators can produce highly effective predictive models.
