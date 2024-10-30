import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Loading the oston housing dataset
# Banco de dados de precos de casas em boston
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(
    data_url,
    sep=r'\s+', 
    skiprows=22,
    header=None
)

print(raw_df.head())

data = np.hstack(
    [raw_df.values[::2, :],
     raw_df.values[1::2, :2]]
)
target = raw_df.values[1::2, 2]

X = data
y = target

# Perform univariate feature selection
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)
feature_names = np.array(
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    dtype=object
)
selected_features = feature_names[selected_indices]

# Get the feature scores
scores = selector.scores_

# Plot the feature scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), scores, tick_label=feature_names)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Scores')
plt.title('Univariate Feature Selection: Feature Scores')
plt.show()

print("Selected Features:")
print(selected_features)