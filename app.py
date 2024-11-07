import openai
import streamlit as st
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
#from google.colab import drive
#drive.mount('/content/drive')

# Load environment variables from a .env file
load_dotenv()

# Streamlit UI
st.title("Shafie Fiqh Ruling Finder")

# Input for user query
query = st.text_area("Enter your question related to Shafie Fiqh rulings:")

# Define the path to the FAISS index
#output_file_path = "/content/drive/MyDrive/UBD/PhD/Embeddings/index.faiss"
import requests

url = "https://storage.cloud.google.com/juristic_v1/index.faiss"  # Public URL of your file
response = requests.get(url, allow_redirects=True)

# Save the file to a local path
with open('/content/index.faiss', 'wb') as f:
    f.write(response.content)

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Encode the query
    query_embedding = model.encode([query])

    # Read FAISS index
    index = faiss.read_index("/content/index.faiss")
    #index = faiss.read_index(output_file_path)
    # Search for similar chunks
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Retrieve the most relevant chunks
    relevant_chunks = [all_chunks[idx] for idx in indices[0]]
    return relevant_chunks

# Function to generate response
def generate_response(query, top_k=3):
    try:
        # Retrieve relevant context
        context = retrieve_relevant_chunks(query, top_k)
        context_text = ' '.join(context)

        # Generate response using GPT-4o
        response = openai.ChatCompletion.create(
            model="gpt-4-2024-08-06",
            messages=[
                {"role": "system", "content": "1. Break down the item into step-by-step actions or processes. 2. Determine whether each action or process is permissible or not using the context provided. 3. Use rulings from Shafie fiqh school that is provided as the context. 4. Provide the dalil both naqli and aqli based on the contexts provided. 5. Explore any counter arguments or any known controversy towards your answer using the context provided. 6. Address and acknowledge those counter arguments and controversy, strictly only when there is any significant counter arguments or controversy. If you think they are valid and correct, please say so. 7. Be careful when providing all the evidences (quran, sunnah, and qaul from scholars). 8. Always solely use the context provided. 9. Don't derive answer on your own, use only the context provided. 10. Always provide final answer, permissible or not."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

# Button to get the response
if st.button("Get Ruling"):
    if query:
        response = generate_response(query)
        st.subheader("Response")
        st.write(response)
    else:
        st.write("Please enter a question.")
