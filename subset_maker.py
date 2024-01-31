# Program scope: create a vector database of plaintext from websites
#                chunk said text and store through lang chain FAISS which requires
#                Data representation as vectors with integer header.

# It's already implemented in langchain for the background:
# https://github.com/facebookresearch/faiss

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
import nest_asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

nest_asyncio.apply()
def convert_to_FAISS(articles: list,**kwargs):
    # Scrapes the blogs above
    loader = AsyncChromiumLoader(articles)
    docs = loader.load()
    # Converts HTML to plain text
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=kwargs['chunk_size'] if 'chunk_size' in kwargs else 100,
                                          chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(docs_transformed)
    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(chunked_documents,
                              HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

    # Connect query to FAISS index using a retriever
    # Grabs the top 5 most similar results
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': (kwargs['similarity'] if 'similarity' in kwargs else 5)}
    )
    return db, retriever


def test_db(query: str, db: FAISS, **kwargs):
    outputs = db.similarity_search(query)
    print(outputs[0].page_content)


# The minimum requirement for performing 4-bit LLM model inference with LMDeploy on NVIDIA graphics cards is sm80,
# which includes models such as the A10, A100, and Geforce RTX 30/40 series.
def gpu_compatibility_bf16():
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # Activate 4-bit precision base model loading
    use_4bit = True
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("The GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
