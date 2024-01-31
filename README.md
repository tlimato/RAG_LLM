# main.py

This repository demonstrates the usage of LangChain, a library for constructing language model chains, by combining a text generation pipeline with a retrieval-augmented generation (RAG) system.

## Usage

To run the provided code:

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Execute the example script:

    ```bash
    python main.py
    ```

This script showcases the creation of a language model chain using LangChain, integrating a Hugging Face transformer model for text generation and a retrieval system based on FAISS.

## Code Overview

- **Bits & Bytes Configurations**: Configure 4-bit precision base model loading for efficient training.

- **Model Initialization**: Load a pre-trained Hugging Face transformer model and tokenizer.

- **Operational Functions**: Define an inference pipeline for text generation based on the loaded model and tokenizer.

- **Prompt Template**: Construct a template for generating prompts, providing context and questions.

- **LangChain Initialization**: Create a LangChain instance using the defined inference pipeline and prompt template.

- **RAG System Creation**: Utilize LangChain to build a Retrieval-Augmented Generation (RAG) system.

- **Invoke the RAG Chain**: Demonstrate the usage of the RAG system by invoking it with a sample question.

## Additional Information

- **Articles**: A set of Python programming-related articles is used for context retrieval.

- **Customization**: Modify the code to adapt the prompt template, replace the pre-trained model, and customize the retrieval system based on specific requirements.

## Dependencies

- [LangChain](https://github.com/langchain/langchain)
- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)




# LangChain Vector Database Creation

This repository demonstrates the creation of a vector database of plaintext from websites using LangChain. The program's scope is to scrape articles, convert HTML to plain text, chunk the text, and store it through LangChain's FAISS vector store.

## Implementation Overview

### Dependencies
- [LangChain](https://github.com/langchain/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [AsyncChromiumLoader](https://github.com/langchain/langchain/blob/main/langchain/document_loaders.py)
- [Html2TextTransformer](https://github.com/langchain/langchain/blob/main/langchain/document_transformers.py)
- [CharacterTextSplitter](https://github.com/langchain/langchain/blob/main/langchain/text_splitter.py)
- [HuggingFaceEmbeddings](https://github.com/langchain/langchain-community/blob/main/langchain_community/embeddings.py)

### Workflow

1. **Web Scraping**: Utilize `AsyncChromiumLoader` to asynchronously load articles from provided URLs.

2. **HTML to Text Transformation**: Convert HTML documents to plain text using `Html2TextTransformer`.

3. **Text Chunking**: Employ `CharacterTextSplitter` to chunk text into specified sizes.

4. **Vector Database Creation**: Load chunked documents into the FAISS index with embeddings obtained from `HuggingFaceEmbeddings`.

5. **Retrieval System Setup**: Connect queries to the FAISS index using a retriever to retrieve the top similar results.

## Usage

1. Install the required packages (redundant):

    ```bash
    pip install -r requirements.txt
    ```

2. Execute the vector database creation script:

    ```bash
    python subset.py
    ```

## Additional Notes

- The script provides GPU compatibility checks for 4-bit LLM model inference with LMDeploy on NVIDIA graphics cards.

- Modify the code to include your specific URLs and adjust parameters as needed.

- Ensure the compatibility of your GPU for bfloat16 to accelerate training.

Feel free to customize and adapt the code for your specific use case. If you have any questions or feedback, please don't hesitate to reach out.

Happy vector database creation with LangChain!


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to explore and adapt the code for your language modeling needs! If you have any questions or feedback, please don't hesitate to contact us.
