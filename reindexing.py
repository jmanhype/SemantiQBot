import os
import pinecone
from pinecone_index import PineconeIndex
from clustering import cluster_rollups
from sparse_priming import sparse_priming

class ReindexingEvent:
    def __init__(self):
        self.index_name = "my_index"
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        self.pinecone_client = pinecone.Client(api_key=api_key)
        self.index = PineconeIndex(self.index_name, api_key=api_key)

    def run(self, rollups):
        # Step 1: Delete all existing documents in the Pinecone index
        self.index.delete_all_documents()

        # Step 2: Cluster the roll-ups based on semantic similarity
        clusters = cluster_rollups(rollups)

        # Step 3: Create conjunctions for the clusters and generate new KB articles
        new_documents = []
        for cluster in clusters:
            conjunctions = sparse_priming(cluster)
            for conjunction in conjunctions:
                new_document = {'text': conjunction, 'cluster_id': cluster[0].cluster_id}
                new_documents.append(new_document)

        # Step 4: Add the new documents to the Pinecone index
        self.index.add_documents(new_documents)
