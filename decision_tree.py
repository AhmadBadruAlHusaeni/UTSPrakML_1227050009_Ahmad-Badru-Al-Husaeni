
# %%
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

# %%
# Load Dataset
df = pd.read_csv("citrus.csv")
print(df.head())

# %%
# Visualisasi Distribusi Kelas
sns.countplot(x='name', data=df)
plt.title("Distribusi Kelas: Orange vs Grapefruit")
plt.show()

# %%
# Visualisasi Data (pairplot)
sns.pairplot(df, hue='name', palette='Set1')
plt.show()

# %%
# Pisahkan Fitur dan Label
X = df.drop('name', axis=1)
y = df['name']

# %%
# Split Data Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Jumlah data latih: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")

# %%
# Training Model Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# %%
# Prediksi dan Evaluasi Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# %%
# Visualisasi Struktur Tree
plt.figure(figsize=(25, 15))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Visualisasi Pohon Keputusan")
plt.show()

# %%
# Prediksi Buah Baru
sample = pd.DataFrame([{
    'diameter': 4.8,
    'weight': 170,
    'red': 140,
    'green': 65,
    'blue': 25
}])

sample = sample[X.columns]
prediction = model.predict(sample)
print("Prediksi buah baru:", prediction[0])
