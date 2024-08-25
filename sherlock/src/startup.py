import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=3600.0)

documents = SimpleDirectoryReader("src/Data").load_data()

persist_directory = './Index'
if(not os.path.exists(persist_directory)):
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_directory)
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    index = load_index_from_storage(storage_context)

chat_engine = index.as_chat_engine()

# chatting logic
user_input = ""
while(user_input != "bye"):
    user_input = input("you: ")
    bot_output = chat_engine.chat(user_input)
    print(f"bot: {bot_output}")