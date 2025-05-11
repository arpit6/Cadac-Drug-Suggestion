from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os

model_name = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

base_dir = os.path.dirname(os.path.abspath(__file__))

data_set = os.path.join(base_dir, 'data_set.csv')
data_embeddings = os.path.join(base_dir, 'embeddings')

def create_faiss_embeddings_csv():
    loader = CSVLoader(data_set,autodetect_encoding=True)
    data = loader.load()
    vectorstore = FAISS.from_documents(data,hf)
    serialize_vector = vectorstore.serialize_to_bytes()
    with open(data_embeddings, 'wb') as f:
        f.write(serialize_vector)

create_faiss_embeddings_csv()