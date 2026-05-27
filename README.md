# telecom-customer-churn


1. ### Description
A machine learning project that predicts customer churn for a marketing agency using a logistic regression model built from scratch in NumPy — implementing the sigmoid function, cross-entropy cost function, and gradient descent manually, without using scikit-learn's LogisticRegression API.
The project covers the full data science pipeline: business understanding → exploratory data analysis → feature engineering → model training → evaluation → prediction on unseen customer data.



2. ### Technologies and tools
Python · NumPy · pandas · matplotlib · seaborn · scikit-learn (metrics only) · Google Colab · Git · GitHub


3. ### Business problem
A marketing agency produces ads for client websites and has noticed significant customer churn. Account managers are currently assigned randomly — meaning high-risk customers receive no more attention than low-risk ones.
The agency needs a model that identifies which customers are most likely to churn, so they can assign account managers proactively to at-risk clients before they leave.

"It costs significantly more to acquire a new client than to retain an existing one. Random account manager assignment wastes retention resources on low-risk customers."

4. ### Solution pipeline

1. Business problem definition
2. Exploratory data analysis (EDA)
3. Feature engineering and preprocessing (StandardScaler)
4. Train/test split — 80/20, stratified
5. Logistic regression built from scratch in NumPy
6. Model evaluation with full classification metrics
7. Prediction on new unlabelled customer data
8. Business interpretation and next steps


5. ### Main business insights

5.1 Churn is the minority class
Approximately 14% of customers churned in the historical dataset. This class imbalance is critical — a naive model predicting "no churn" for every customer would achieve 86% accuracy while being completely useless for the business.
This is why accuracy alone is a misleading metric for this problem, and why Precision, Recall, F1, and ROC-AUC are the correct evaluation framework.


5.2 Number of websites is a strong churn signal
Customers using the service across more websites show lower churn rates — deeper product engagement correlates with retention. This suggests the agency should prioritise onboarding clients onto multiple website integrations early in the relationship.


5.3 Total purchase history matters
Customers with lower total ad spend show higher churn rates. Smaller or newer clients may need earlier and more proactive account manager intervention before they disengage.


5.4 Random account manager assignment shows no protective effect
The current random assignment policy does not show a statistically meaningful reduction in churn. This directly validates the business case for model-driven assignment — the existing approach is not working.


5.5 Feature correlations
No strong multicollinearity between numerical features, confirming that each variable contributes independent predictive signal to the model.


6. ### Modelling
Why from scratch?
Rather than calling LogisticRegression().fit(), this project implements the full algorithm manually in NumPy — including the sigmoid function, cross-entropy cost function, gradient descent optimiser, and decision threshold. This demonstrates mathematical understanding of how logistic regression actually works, not just API usage.
Training configuration

Features used: Age, Total_Purchase, Account_Manager, Years, Num_Sites
Dropped: Name, Onboard_date, Location, Company (non-predictive identifiers)
Preprocessing: StandardScaler (logistic regression is sensitive to feature scale)
Intercept: Added manually as a bias term
Optimiser: Gradient descent — learning rate 0.001, 10,000 iterations

### Interpretation
The model correctly identifies 98% of customers who will not churn — very strong performance on the majority class. However, it captures only 47% of actual churners, missing more than half of at-risk customers.
This low recall on the minority class is a direct consequence of class imbalance (86% no churn, 14% churn) and the linear decision boundary of logistic regression. The model


## 🚀 Live Demo
👉 https://telecom-customer-churn-5vmapsga4noyczl2ukvngu.streamlit.app
