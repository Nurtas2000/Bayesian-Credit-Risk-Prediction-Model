import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             confusion_matrix, classification_report)
import shap

# Генерация данных (используем ваш датасет)
data = pd.read_csv("credit_data.csv")

# Полный код модели
class BayesianCreditRiskModel:
    def __init__(self, n_chains=4, n_samples=2000, n_tune=1000):
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.model = None
        self.trace = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df):
        # Feature engineering
        df['loan_percent_income'] = df['loan_amount'] / (df['salary'] + 1e-6)
        df = pd.get_dummies(df, columns=['home_ownership'], drop_first=True)
        
        features = ['age', 'salary', 'credit_score', 
                   'loan_percent_income', 'years_employed',
                   'home_ownership_MORTGAGE', 'home_ownership_OWN']
        
        X = df[features]
        y = df['loan_status'].values
        self.feature_names = features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, X, y):
        with pm.Model() as model:
            # Иерархические priors
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)
            
            # Коэффициенты
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, 
                           shape=X.shape[1])
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            
            # Линейная комбинация
            mu = alpha + pm.math.dot(X, beta)
            theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', p=theta, observed=y)
            
            self.model = model
        return model
    
    def fit_model(self, X, y):
        with self.model:
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=0.9,
                return_inferencedata=True)
        return self.trace
    
    def evaluate(self, X_test, y_test):
        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.trace,
                var_names=['theta'],
                random_seed=42)
            
        pred_probs = ppc['theta'].mean(axis=0)
        roc_auc = roc_auc_score(y_test, pred_probs)
        pr_auc = average_precision_score(y_test, pred_probs)
        
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"PR AUC: {pr_auc:.3f}")
        
        # Матрица ошибок
        y_pred = (pred_probs > 0.5).astype(int)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Кривые качества
        self.plot_roc_pr(y_test, pred_probs)
        return roc_auc, pr_auc
    
    def plot_roc_pr(self, y_true, y_pred):
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        plt.figure(figsize=(12, 5))
        
        # ROC curve
        plt.subplot(121)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_true, y_pred):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        
        # PR curve
        plt.subplot(122)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision, label=f"PR (AUC = {average_precision_score(y_true, y_pred):.2f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_features(self):
        # SHAP значения
        background = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_names))
        explainer = shap.Explainer(
            model=lambda X: 1 / (1 + np.exp(-(X @ background.T))),
            masker=background.mean(axis=0).reshape(1, -1),
            feature_names=self.feature_names)
        
        shap_values = explainer(background.mean(axis=0).reshape(1, -1))
        
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title("Feature Impact on Prediction")
        plt.tight_layout()
        plt.show()

# Запуск полного анализа
model = BayesianCreditRiskModel()
X_train, X_test, y_train, y_test = model.preprocess_data(data)
model.build_model(X_train, y_train)
trace = model.fit_model(X_train, y_train)

print("\n=== Model Evaluation ===")
roc_auc, pr_auc = model.evaluate(X_test, y_test)

print("\n=== Feature Analysis ===")
model.analyze_features()

print("\n=== Bayesian Summary ===")
print(pm.summary(trace, var_names=['alpha', 'beta']))
