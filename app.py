import streamlit as st
import wikipediaapi
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def load_models():
    """Loads SBERT and Flan-T5 models once and caches them."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # SBERT for retrieval
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32)
    return embedding_model, tokenizer, model

embedding_model, tokenizer, model = load_models()

dimension = 384  # SBERT output dimension
index = faiss.IndexFlatL2(dimension)
stored_chunks = []  # Stores text chunks alongside embeddings

def scrape_wikipedia_text(topic):
    """Fetch Wikipedia text with a proper user-agent."""
    user_agent = "Wikipedia-RAG/1.0 (https://github.com/your-repo-name)"  # Random GitHub link to appease Wikipedia
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")
    page = wiki.page(topic)

    return page.text if page.exists() else None

def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def embed_chunks(chunks):
    """Converts text chunks into embeddings using SBERT."""
    return embedding_model.encode(chunks)

def store_embeddings(topic):
    """Scrapes Wikipedia, chunks text, embeds it, and stores in FAISS."""
    text = scrape_wikipedia_text(topic)
    if text:
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        embeddings_np = np.array(embeddings).astype("float32")
        index.add(embeddings_np)
        stored_chunks.extend(chunks)
        return len(chunks)
    return 0

def retrieve_chunks(query, top_k=3):
    """Finds the most relevant text chunks for a given query using FAISS."""
    if index.ntotal == 0:
        return ["No data available. Please scrape and store a Wikipedia topic first."]

    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    retrieved_texts = [stored_chunks[i] for i in indices[0] if i < len(stored_chunks)]

    return retrieved_texts if retrieved_texts else ["No relevant information found."]

def generate_response(query):
    """Generates a response using Flan-T5 with retrieved Wikipedia context."""
    retrieved_texts = retrieve_chunks(query)

    if "No data available" in retrieved_texts or "No relevant information found." in retrieved_texts:
        return retrieved_texts[0]  # Return the error message instead of running the model

    context = "\n\n".join(retrieved_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=500)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


st.title("DIY Wikipedia RAG")

# First, the user provides any number of Wikipedia article titles to build the database for RAG

topic = st.text_input("Enter a Wikipedia page name:")
if st.button("Scrape and Store Data"):
    num_chunks = store_embeddings(topic)
    if num_chunks:
        st.success(f"Stored {num_chunks} chunks for topic: {topic}")
    else:
        st.error("Wikipedia page not found!")

# Then, the user can make a query into the vector database they have scraped from Wikipedia

query = st.text_input("Give a prompt related to the pages you scraped:")
if st.button("Get Answer"):
    response = generate_response(query)
    st.write(response)
