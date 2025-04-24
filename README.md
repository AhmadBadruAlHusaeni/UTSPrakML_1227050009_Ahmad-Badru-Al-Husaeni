# UTSPrakML_1227050009_Ahmad-Badru-Al-Husaeni
#Decision Tree

 1. Pengumpulan dan Persiapan Data
Dataset yang digunakan diunduh dari Kaggle: 🔗 https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit

Langkah awal:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("citrus.csv")  # Ganti sesuai nama file dalam zip
print(df.head())
print(df.info())

 2. Eksplorasi Data dan Visualisasi
Melihat statistik ringkasan:
print(df.describe())

Visualisasi sebaran fitur terhadap kelas:
sns.pairplot(df, hue='name', palette='Set1')
plt.show()

 3. Pembagian Data (Training dan Testing)
Pisahkan fitur dan label:
X = df.drop('name', axis=1)
y = df['name']

 4. Membangun Model Decision Tree
Gunakan metode DecisionTreeClassifier:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report  # Import classification_report here

model = DecisionTreeClassifier(
    criterion='entropy',            
    random_state=42             
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

 5. Evaluasi Model
 Classification Report:
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

 Confusion Matrix (Visualisasi):
 from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

 6. Visualisasi Decision Tree
Visualisasi struktur pohon:
from sklearn import tree

plt.figure(figsize=(25, 20))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Visualisasi Pohon Keputusan")
plt.show()

 7. Uji Coba Data Baru
Uji model dengan data uji manual:
sample = pd.DataFrame([{
    'diameter': 4.7,
    'weight': 180,
    'red': 145,
    'green': 68,
    'blue': 22
}])

sample = sample[X.columns]
prediction = model.predict(sample)
print("Prediksi buah:", prediction[0])
