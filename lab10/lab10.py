import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import time
from sklearn.metrics import roc_curve


X,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Wygenerowane dane')
plt.xlabel('Cechy 1')
plt.ylabel('Cechy 2')
plt.show()

models = {}

models['Naive Bayes'] = GaussianNB()
models['QDA'] = QuadraticDiscriminantAnalysis()
models['K-Nearest Neighbor'] = KNeighborsClassifier()
models['Support Vector Machines'] = SVC(probability=True)
models['Decision Trees'] = DecisionTreeClassifier()

results = {name: {'accuracy': [], 'recall': [], 'precision': [], 'f1': [], 'roc_auc': [], 'train_time': [], 'test_time': []} for name in models.keys()}

for i in range (100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        train_time = end - start
        results[name]['train_time'].append(train_time)

        # Measure testing time
        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        test_time = end - start
        results[name]['test_time'].append(test_time)

        results[name]['accuracy'].append(metrics.accuracy_score(y_test, y_pred))
        results[name]['recall'].append(metrics.recall_score(y_test, y_pred))
        results[name]['precision'].append(metrics.precision_score(y_test, y_pred))
        results[name]['f1'].append(metrics.f1_score(y_test, y_pred))
        results[name]['roc_auc'].append(metrics.roc_auc_score(y_test, y_pred) if y_pred is not None else float('nan'))

summary = {name: {metric: (np.mean(values)) for metric, values in metrics.items()} for name, metrics in results.items()}

df_model = pd.DataFrame({
    'Classifier': list(summary.keys()),
    'Accuracy Mean': [summary[name]['accuracy'] for name in summary.keys()],

    'Recall Mean': [summary[name]['recall'] for name in summary.keys()],

    'Precision Mean': [summary[name]['precision'] for name in summary.keys()],

    'F1 Mean': [summary[name]['f1'] for name in summary.keys()],

    'ROC AUC Mean': [summary[name]['roc_auc'] for name in summary.keys()],

    'Train Time ': [summary[name]['train_time'] for name in summary.keys()],

    'Test Time ': [summary[name]['test_time'] for name in summary.keys()]
})
print(df_model)

df_model_transposed = df_model.set_index('Classifier').T

ax = df_model_transposed.plot.bar(figsize=(12, 8))

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 10})


plt.show()


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50)
plt.title('Oczekiwane')

plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.title('obliczone')

plt.show()

correct_indices = np.where(y_test == y_pred)[0]
incorrect_indices = np.where(y_test != y_pred)[0]
plt.scatter(X_test[correct_indices, 0], X_test[correct_indices, 1], c='green', edgecolor='k', s=50, label='Correct')
plt.scatter(X_test[incorrect_indices, 0], X_test[incorrect_indices, 1], c='red', edgecolor='k', s=50, label='Incorrect')
plt.title('Roznice')

plt.legend()

plt.tight_layout()
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Rysowanie krzywej ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC krzywa')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

