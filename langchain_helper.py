import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

access_token = os.environ.get("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)


file_path = (
    "/Users/apple/Documents/cafeschennai.pdf"
)

loaders = PyPDFLoader(file_path)

data = loaders.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)

# Create global embeddings and retrievers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create global vector index from all documents
vectorindex = FAISS.from_documents(docs, embeddings)
embedding_retriever = vectorindex.as_retriever(search_kwargs={"k": 15})  # Increased k

# Create global BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 15  # Increased k

# Create global hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, embedding_retriever],
    weights=[0.5, 0.5]
)

# Improved prompt template
cafe_prompt_template = PromptTemplate.from_template("""You are a helpful assistant that extracts cafe information from the provided context.

IMPORTANT INSTRUCTIONS:
1. Only list cafes that are specifically located in {location}
2. Extract ALL cafes mentioned in the context that are in {location}
3. Look carefully through the entire context for multiple cafe listings
4. Do NOT stop after finding the first cafe - continue reading to find ALL cafes

Based on the context provided, find ALL cafes that are located in {location}. Look for cafes where the address, description, or location specifically mentions {location}.

For EACH cafe that is actually located in {location}, format the response as:

Cafe Name: [Name]
Description: [Description from context]
Address: 
[Address from context]
Popular Dishes: 
[Popular dishes/items if mentioned]

---

If any field is not available in the context, write "Not specified".

CRITICAL: 
- Extract ALL cafes from the context, not just the first one
- If you find multiple cafes, list them all using the same format
- Separate each cafe with "---"
- If you cannot find any cafes specifically located in {location}, respond with "No cafes found in {location}"

Context:
{context}

Question: {question}
Answer:
""")

general_prompt_template = PromptTemplate.from_template("""You are a helpful assistant that answers questions based on the provided context about cafes and restaurants.

Provide a clear, accurate, and helpful answer based on the context provided. If the information is not available in the context, say so clearly.

Context:
{context}

Question: {question}
Answer:
""")

def get_cafe_information(location):

    # First, filter documents that contain the location
    location_variations = [
        location.lower(),
        location.replace(" ", "").lower(),
        location.replace("-", " ").lower(),
        location.replace("_", " ").lower(),
        location.title(),
        location.upper()
    ]

    filtered_docs = []
    for doc in docs:
        doc_content_lower = doc.page_content.lower()
        if any(var in doc_content_lower for var in location_variations):
            filtered_docs.append(doc)

    print(f"Found {len(filtered_docs)} documents containing '{location}'")

    if not filtered_docs:
        return [("No results", f"No documents found containing information about {location}")]

    # Create new retrievers with filtered documents
    try:
        # Create BM25 retriever with filtered docs
        bm25_filtered = BM25Retriever.from_documents(filtered_docs)
        bm25_filtered.k = 10  # Increased

        # Create vector retriever with filtered docs
        vector_filtered = FAISS.from_documents(filtered_docs, embeddings)
        vector_retriever_filtered = vector_filtered.as_retriever(search_kwargs={"k": 10})

        # Create ensemble retriever
        ensemble_filtered = EnsembleRetriever(
            retrievers=[bm25_filtered, vector_retriever_filtered],
            weights=[0.5, 0.5]
        )

        # Enhanced query
        query = f"List ALL cafes in {location}. Find every cafe mentioned in the documents that is located in {location}. Include their names, descriptions, addresses, and contact information for each cafe."

        chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=ensemble_filtered,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": cafe_prompt_template},
        )

        result = chain.invoke({
            "question": query,
            "chat_history": [],
            "location": location
        })

        answer_text = result["answer"]

        print(f"LLM Response: {answer_text}")
        print("=" * 50)

        # Improved parsing - split by "---" first, then by double newlines
        cafes = []

        # Split by "---" separator first
        if "---" in answer_text:
            sections = answer_text.split("---")
        else:
            # Fallback to double newline split
            sections = re.split(r'\n\s*\n', answer_text)

        for section in sections:
            if not section.strip():
                continue

            cafe_data = {
                "name": "", "description": "", "address": "",
                "popular_dishes": ""
            }

            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Cafe Name:"):
                    cafe_data["name"] = line.replace("Cafe Name:", "").strip()
                elif line.startswith("Description:"):
                    cafe_data["description"] = line.replace("Description:", "").strip()
                elif line.startswith("Address:"):
                    cafe_data["address"] = line.replace("Address:", "").strip()
                elif line.startswith("Popular Dishes:"):
                    cafe_data["popular_dishes"] = line.replace("Popular Dishes:", "").strip()

            if cafe_data["name"]:
                full_description = []
                if cafe_data["description"]:
                    full_description.append(cafe_data["description"])
                if cafe_data["address"]:
                    full_description.append(f"Address:\n{cafe_data['address']}")
                if cafe_data["popular_dishes"]:
                    full_description.append(f"Popular Dishes:\n{cafe_data['popular_dishes']}")

                cafes.append((cafe_data["name"], "\n\n".join(full_description)))

        return cafes if cafes else [("Raw Response", answer_text)]

    except Exception as e:
        print(f"Error in improved approach: {e}")
        return [("Error", f"Error processing cafes for {location}: {str(e)}")]

def q_and_a(question, location):
    query = question

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=hybrid_retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": general_prompt_template},
    )

    chat_history = []

    result = chain.invoke({"question": query, "chat_history": chat_history, "location": location})
    return result

