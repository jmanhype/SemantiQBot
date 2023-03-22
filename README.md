# SemantiQBot

This project is a chatbot that uses Pinecone, a vector search engine, and GPT-3, a powerful language model, to answer user queries. The chatbot clusters text data based on semantic similarity, creates conjunctions for the clusters, generates new knowledge base (KB) articles, and adds them to the Pinecone index. It then uses GPT-3 to generate a response based on the most relevant KB article.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Pinecone API key
- OpenAI API key

### Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/semantic-chatbot.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Set up environment variables for Pinecone and OpenAI API keys:

```
export PINECONE_API_KEY=your_pinecone_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### Usage

1. Run the main script:

```
python main.py
```

2. Interact with the chatbot by typing your queries:

```
User: What is the capital of France?
Chatbot: The capital of France is Paris.
```

## Project Structure

- `main.py`: The main script that sets up the Pinecone index, initializes data structures, and runs the chatbot.
- `clustering.py`: Contains the `Rollup`, `Chunk`, `Cluster`, and `Clustering` classes for representing and clustering text data.
- `gpt3.py`: Contains the `generate_response` function that uses the OpenAI API to generate a response from GPT-3.
- `pinecone_index.py`: Contains the `PineconeIndex` class for managing the Pinecone index.
- `reindexing_event.py`: Contains the `ReindexingEvent` class for running the reindexing process on a set of roll-ups.
- `sparse_priming.py`: Contains the `create_conjunctions` function for creating conjunctions for a given cluster of roll-ups.