import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings


def create_faiss_embeddings_csv():
    """
    Load CSV data, compute embeddings using a HuggingFace model,
    and serialize the resulting FAISS vector store to disk.
    """
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(base_dir, 'data_set.csv')
    embeddings_path = os.path.join(base_dir, 'embeddings')

    loader = CSVLoader(data_set_path, autodetect_encoding=True)
    documents = loader.load()

    vectorstore = FAISS.from_documents(documents, embeddings)
    serialized = vectorstore.serialize_to_bytes()

    with open(embeddings_path, 'wb') as f:
        f.write(serialized)


if __name__ == "__main__":
    create_faiss_embeddings_csv()