import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("spotify.csv")

print(df.head())

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.dropna(subset=[
    "track_name",
    "track_artist",
    "track_album_name"
])

genre_counts = df["playlist_genre"].value_counts()

print("\nSongs per Genre:")
print(genre_counts)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="playlist_genre")

plt.title("Distribution of Spotify Playlist Genres")
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(df["track_popularity"], bins=30, kde=True)

plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")

plt.show()

audio_features = [
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "speechiness"
]

df[audio_features].hist(figsize=(14, 10), bins=30)

plt.suptitle("Distribution of Spotify Audio Features")
plt.tight_layout()

plt.show()

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

numerical_df = df.select_dtypes(include=np.number)

correlation_matrix = numerical_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Spotify Features")
plt.tight_layout()
plt.show()

genre_features = df.groupby("playlist_genre")[
    ["danceability", "energy", "acousticness", "valence", "speechiness"]
].mean()

print("\nAverage Audio Features by Genre:")
print(genre_features)

genre_features.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Average Audio Features Across Music Genres")
plt.xlabel("Music Genre")
plt.ylabel("Average Feature Value")
plt.xticks(rotation=45)
plt.legend(title="Audio Features")
plt.tight_layout()
plt.show()

playlist_counts = df["playlist_name"].value_counts()

print("\nTop 10 Most Common Playlists:")
print(playlist_counts.head(10))

plt.figure(figsize=(12, 7))

sns.barplot(
    x=playlist_counts.head(10).values,
    y=playlist_counts.head(10).index
)

plt.title("Top 10 Playlists by Number of Songs")
plt.xlabel("Number of Songs")
plt.ylabel("Playlist Name")

plt.tight_layout()
plt.show()

features = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X = df[features]

print("\nSelected Features:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardized Data:")
print(X_scaled[:5])

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)

plt.show()

silhouette_scores = []

k_values = [3, 4, 5, 6]

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=3000,
        random_state=42
    )

    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.grid(True)

plt.show()

kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster_5"] = kmeans_5.fit_predict(X_scaled)

genre_cluster = pd.crosstab(
    df["cluster_5"],
    df["playlist_genre"],
    normalize="index"
) * 100

print("\nCluster Distribution Across Genres:")
print(genre_cluster)

plt.figure(figsize=(10, 6))

sns.heatmap(
    genre_cluster,
    annot=True,
    cmap="YlGnBu",
    fmt=".1f"
)

plt.title("Playlist Genre Distribution Within Each Cluster")
plt.xlabel("Playlist Genre")
plt.ylabel("Cluster")

plt.show()

for cluster in sorted(df["cluster_5"].unique()):

    print(f"\nTop Playlists in Cluster {cluster}:")

    top_playlists = df[df["cluster_5"] == cluster][
        "playlist_name"
    ].value_counts().head(5)

    print(top_playlists)

print("\nNumber of Songs in Each Cluster (K=5):")
print(df["cluster_5"].value_counts().sort_index())

print("\nAverage Audio Features (K=5):")
cluster_5_profile = df.groupby("cluster_5")[features].mean()
print(cluster_5_profile.round(2))

print("\nGenre Distribution (K=5):")
genre_distribution_5 = pd.crosstab(
    df["cluster_5"],
    df["playlist_genre"],
    normalize="index"
) * 100

print(genre_distribution_5.round(1))

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="danceability",
    y="energy",
    hue="cluster_5",
    palette="viridis",
    alpha=0.6
)

plt.title("Spotify Song Clusters (K=5): Danceability vs Energy")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.tight_layout()
plt.show()

print("\nRandom Sample of Songs in Each Cluster (K=5):")

for cluster in sorted(df["cluster_5"].unique()):
    print(f"\nCluster {cluster}:")

    songs = df[df["cluster_5"] == cluster].sample(
        n=10,
        random_state=42
    )

    print(songs[
        ["track_name", "track_artist", "playlist_genre"]
    ].to_string(index=False))

cluster_means = df.groupby("cluster_5")[
    ["danceability", "energy", "liveness", "valence", "tempo"]
].mean()

genre_distribution = pd.crosstab(
    df["cluster_5"],
    df["playlist_genre"],
    normalize="index"
) * 100

cluster_summary = cluster_means.copy()

cluster_summary["song_count"] = df["cluster_5"].value_counts().sort_index()
cluster_summary["dominant_genre"] = genre_distribution.idxmax(axis=1)
cluster_summary["genre_percentage"] = genre_distribution.max(axis=1)

print("\nFinal Cluster Interpretation:")
print(cluster_summary.round(2))
