import os
from threading import Timer
from pinecone_index import PineconeIndex
from clustering import Clustering
from sparse_priming import create_conjunctions
from gpt3 import generate_response

# Initialize Pinecone index
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pinecone_index = PineconeIndex(index_name="my-sample-index", api_key=pinecone_api_key)
print(f"Pinecone index: {pinecone_index}")

# Define data structures
kb_articles = {}  # dictionary of KB articles
rollups = [{'id': '1', 'text': 'What is the capital of France?'},
           {'id': '2', 'text': 'How many countries are there in the European Union?'},
           {'id': '3', 'text': 'What is the largest mammal on Earth?'}
          ]  # list of roll-ups
chunks = []  # list of chunks

# Embed and store the initial rollups in the Pinecone index
embeddings = [pinecone_index.embed_text(rollup['text']) for rollup in rollups]
pinecone_index.add_documents({rollup['id']: vec.tolist() for rollup, vec in zip(rollups, embeddings)})

print("Fetching added documents using fetch method:")
fetch_response = pinecone_index.index.fetch(ids=['1', '2', '3'])
print("Fetched data:", fetch_response.received_data)

print("Testing retrieve_document:")
for doc_id in ['1', '2', '3']:
    print(f"Document {doc_id}:", pinecone_index.retrieve_document(doc_id))

def reindex_periodically():
    reindex()
    timer = Timer(86400, reindex_periodically)  # 24 hours
    timer.start()

def reindex():
    global kb_articles, chunks, rollups
    # Step 1: Cluster the roll-ups based on semantic similarity
    clustering = Clustering(index_name="my-sample-index", pinecone_index=pinecone_index, rollups=rollups)
    clusters = clustering.cluster_rollups(threshold=0.5)

    if not clusters:
        print("No clusters found. Not updating index.")
        return

    # Step 2: Delete all existing documents in the Pinecone index
    pinecone_index.delete_all_documents()

    # Step 3: Create conjunctions for the clusters and generate new KB articles
    new_documents = []
    kb_articles.clear()
    for cluster in clusters:
        conjunctions = create_conjunctions(cluster)
        conjunction_text = " AND ".join([f"({rollup.text})" for rollup in cluster])
        new_document = {'text': conjunction_text, 'cluster_id': cluster[0].cluster_id}
        new_documents.append(new_document)
        answer = generate_response(f"What is {conjunction_text}?")
        kb_articles[cluster[0].cluster_id] = answer

    # Step 4: Add the new documents to the Pinecone index
    if new_documents:  # Check if there are any new_documents to upsert
        embeddings = [pinecone_index.embed_text(new_document['text']) for new_document in new_documents]
        pinecone_index.upsert([(str(new_document['cluster_id']), vec.tolist()) for new_document, vec in zip(new_documents, embeddings)])
    else:
        print("No new documents to upsert")

    # Update chunks
    chunks = clusters

reindex_periodically()

# Define chatbot function
def chatbot(query: str) -> str:
    """
    Process a user query and generate a response.

    Args:
        query: The user's question

    Returns:
        The chatbot's response

    Raises:
        ValueError: If query is empty
        RuntimeError: If search or response generation fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    try:
        # Search Pinecone index for closest KB article
        result = pinecone_index.search(query)

        if not result or not result.get('ids'):
            return "I'm sorry, I couldn't find any relevant information to answer your question."

        closest_article = result['ids'][0]

        # Check if article exists in KB
        if closest_article not in kb_articles:
            return "I'm sorry, I don't have enough information to answer that question yet."

        # Use GPT-3 to generate response
        response = generate_response(kb_articles[closest_article] + " " + query)
        return response
    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question."

# Use chatbot to respond to user queries
while True:
    query = input("User: ")
    response = chatbot(query)
    print("Chatbot:", response)