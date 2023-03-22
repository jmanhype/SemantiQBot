import pinecone
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

class Rollup:
    def __init__(self, text, id):
        self.text = text
        self.id = id
        self.embedding = None

    def to_vector(self, pinecone_index):
        if self.embedding is None:
            document = pinecone_index.retrieve_document(self.id)
            if document:
                self.embedding = np.array(document['values'])
            else:
                return None
        return self.embedding

class Chunk:
    def __init__(self, rollups):
        self.rollups = rollups

class Cluster:
    def __init__(self, rollups):
        self.rollups = rollups

class Clustering:
    def __init__(self, index_name, pinecone_index, rollups=None):
        self.index_name = index_name
        self.rollups = [Rollup(text=rollup['text'], id=rollup['id']) for rollup in rollups] if rollups else []
        self.clusters = []
        self.pinecone_index = pinecone_index

    def load_rollups(self):
        if not self.rollups:
            ids = self.pinecone_index.fetch_ids()
            fetched_data = self.pinecone_index.index.fetch(ids=ids).received_data
            self.rollups = [Rollup(text=rollup['text'], id=id) for id, rollup in fetched_data.items()]

    def cluster_rollups(self, threshold):
        self.load_rollups()
        # Convert rollups to vector form
        vectors = [rollup.to_vector(self.pinecone_index) for rollup in self.rollups]

        # Filter out any None values from the vectors list
        vectors = list(filter(None, vectors))

        if not vectors:  # Check if the vectors list is empty
            print("No rollups found to cluster.")
            return []

        # Calculate cosine similarity matrix
        similarities = cosine_similarity(vectors)

        # Perform clustering based on similarity
        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=threshold)
        clustering.fit(similarities)

        # Create clusters based on clusters
        self.clusters = []
        for cluster_idx in range(clustering.n_clusters_):
            cluster_rollups = [self.rollups[i] for i in range(len(self.rollups)) if clustering.labels_[i] == cluster_idx]
            cluster = Cluster(cluster_rollups)
            self.clusters.append(cluster)

        return self.clusters

    def reindex(self):
        self.cluster_rollups(threshold=0.5)

        # Delete existing rollup index
        pinecone.delete_index(index_name=self.index_name)

        # Upsert rollups to Pinecone index
        embeddings = np.array([rollup.to_vector() for rollup in self.rollups])
        pinecone_index = pinecone.Index(index_name=self.index_name)
        pinecone_index.upsert([rollup.id for rollup in self.rollups], embeddings)
