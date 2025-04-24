# UTSPrakML_1227050009_Ahmad-Badru-Al-Husaeni
# Decision Tree

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

![image](https://github.com/user-attachments/assets/63acdfb1-fc0c-48ad-99eb-71bc9c051de2)

 2. Eksplorasi Data dan Visualisasi
Melihat statistik ringkasan:
print(df.describe())

![image](https://github.com/user-attachments/assets/46739f56-89cf-4f02-8807-07b7fffdb82e)

Visualisasi sebaran fitur terhadap kelas:
sns.pairplot(df, hue='name', palette='Set1')
plt.show()

![image](https://github.com/user-attachments/assets/e3b67768-9151-49b4-b006-81292f1600ec)


 3. Pembagian Data (Training dan Testing)
Pisahkan fitur dan label:
X = df.drop('name', axis=1)
y = df['name']

Bagi menjadi data latih dan data uji (70:30):
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train))  # Jumlah data latih

![image](https://github.com/user-attachments/assets/9f0464d0-ed10-4249-8845-e3080be47808)

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
![image](https://github.com/user-attachments/assets/e8d5bdab-20fa-43f2-9b27-f75576b69189)

 5. Evaluasi Model
 Classification Report:
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

![image](https://github.com/user-attachments/assets/240b647a-23ff-49fe-9cbe-7ca49937770e)

 Confusion Matrix (Visualisasi):
 from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

![image](https://github.com/user-attachments/assets/e5a1b300-cbec-4d87-8701-80083d82fa1a)


 6. Visualisasi Decision Tree
Visualisasi struktur pohon:
from sklearn import tree

plt.figure(figsize=(25, 20))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Visualisasi Pohon Keputusan")
plt.show()

![image](https://github.com/user-attachments/assets/9d18268a-7878-49f7-8f16-fde6404e9ef3)


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

![image](https://github.com/user-attachments/assets/23f2631b-9599-42b6-bf0f-3ea8f919b472)

prediction = model.predict(sample)
print("Prediksi buah:", prediction[0])
