import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop kolom ID dan baris dengan BMI kosong
df = df.drop(columns=["id"])
df = df.dropna(subset=["bmi"])

# Encode kolom kategorikal
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
df_encoded = df.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# PILIH fitur yang lebih relevan (hindari fitur yang terlalu banyak kategori)
selected_features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
X = df_encoded[selected_features]

# Normalisasi fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# REDUKSI DIMENSI: Gunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Coba nilai K dari 2 hingga 10 dan hitung Silhouette Score
sil_scores = []
print("Silhouette Score untuk setiap k:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.2f}")

# Plot Silhouette Score vs K
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.title("Silhouette Score untuk Berbagai Nilai K")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_score_vs_k.png")
plt.show()

# Pilih k terbaik (dengan Silhouette Score tertinggi)
optimal_k = range(2, 11)[sil_scores.index(max(sil_scores))]
print(f"Nilai K optimal berdasarkan Silhouette Score: {optimal_k}")

# Training ulang dengan k optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_encoded["cluster"] = kmeans.fit_predict(X_pca)

# Tambahkan hasil PCA ke data
df_encoded["PCA1"] = X_pca[:, 0]
df_encoded["PCA2"] = X_pca[:, 1]

# Visualisasi cluster berdasarkan PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_encoded, x="PCA1", y="PCA2", hue="cluster", palette="Set2")
plt.title(f"Visualisasi Cluster (K={optimal_k}) dengan PCA")
plt.tight_layout()
plt.savefig("cluster_pca_optimal.png")
plt.show()

# Distribusi stroke per cluster
stroke_cluster = df_encoded.groupby("cluster")["stroke"].value_counts(normalize=True).unstack().fillna(0)

# Bar chart stroke distribution
stroke_cluster.plot(kind="bar", stacked=True, colormap="coolwarm", figsize=(8, 6))
plt.title("Proporsi Pasien Stroke di Tiap Cluster")
plt.xlabel("Cluster")
plt.ylabel("Proporsi")
plt.legend(title="Stroke (0 = Tidak, 1 = Ya)")
plt.tight_layout()
plt.savefig("stroke_distribution_per_cluster.png")
plt.show()

# Tampilkan proporsi stroke
print("Proporsi Pasien Stroke per Cluster:")
print(stroke_cluster.round(2))
