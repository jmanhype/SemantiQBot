import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Union, List, Optional, Tuple
import os

class PineconeIndex:
    """Manages a Pinecone vector index for semantic search."""

    def __init__(self, index_name: str, api_key: str):
        """
        Initialize the Pinecone index.

        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key

        Raises:
            ValueError: If index_name or api_key is empty
        """
        if not index_name:
            raise ValueError("index_name cannot be empty")
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.index_name = index_name
        self.api_key = api_key
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        embedding_dimension = self.model.get_sentence_embedding_dimension()

        # Get environment from env var or use default
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')

        try:
            pinecone.init(api_key=self.api_key, environment=environment)
            # Only create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=embedding_dimension)
            self.index = pinecone.Index(index_name=index_name)
            self.vector_ids = set()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone index: {str(e)}") from e

    def fetch_or_create_embedding(self, document_id, document_text):
        if document_id in self.vector_ids:
            return np.array(self.retrieve_document(document_id))
        else:
            embedding = self.embed_text(document_text)
            self.add_document(document_id, document_text)
            return embedding

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text into a vector representation.

        Args:
            text: The text to embed

        Returns:
            The embedding vector

        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Text cannot be empty")
        return self.model.encode(text)

    def add_document(self, document_id, document):
        print(f"Adding document: {document_id}")
        self.index.upsert(vectors=[(document_id, self.embed_text(document))])
        self.vector_ids.add(document_id)

    def add_documents(self, documents: Dict[str, Union[str, List[float]]]):
        print(f"Adding documents: {documents.keys()}")
        vectors = [(doc_id, vec) for doc_id, vec in documents.items()]
        self.index.upsert(vectors)
        self.vector_ids.update(documents.keys())
        print(f"Current vector IDs: {self.vector_ids}")

    def delete_document(self, document_id):
        self.index.delete(ids=[document_id])
        self.vector_ids.discard(document_id)

    def delete_all_documents(self):
        if self.vector_ids:
            self.index.delete(ids=list(self.vector_ids))
            self.vector_ids.clear()

    def update_document(self, document_id, document):
        self.index.upsert(vectors=[(document_id, self.embed_text(document))])

    def retrieve_document(self, document_id):
        response = self.index.fetch(ids=[document_id])
        print("Pinecone response:", response)
        if response.received_data and document_id in response.received_data:
            return response.received_data[document_id]
        else:
            print(f"Document not found: {document_id}")
            return None

    def search(self, query, k=10):
        query_embedding = self.embed_text(query).tolist()
        results = self.index.query(queries=[query_embedding], top_k=k)
        return {'ids': [r.id for r in results], 'scores': [r.score for r in results], 'received_data': results.received_data}

    def search_documents(self, query, k=10):
        query_embedding = self.embed_text(query)
        results = self.index.query(queries=[query_embedding], top_k=k)
        return [(r[0].id, r[0].score) for r in results]

    def fetch_ids(self):
        return list(self.vector_ids)

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)
