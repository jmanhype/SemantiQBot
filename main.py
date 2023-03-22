from threading import Timer
from pinecone_index import PineconeIndex
from clustering import Clustering
from sparse_priming import create_conjunctions
from gpt3 import generate_response

# Initialize Pinecone index
pinecone_index = PineconeIndex(index_name="my-sample-index", api_key='your-api-key-here')
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
def chatbot(query):
    # Search Pinecone index for closest KB article
    result = pinecone_index.search(query)
    closest_article = result['ids'][0]

    # Use GPT-3 to generate response
    response = generate_response(kb_articles[closest_article] + " " + query)

    return response

# Use chatbot to respond to user queries
while True:
    query = input("User: ")
    response = chatbot(query)
    print("Chatbot:", response)