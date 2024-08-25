# Using Document Summary Index technique for indexing

import os
import sys

import nest_asyncio
nest_asyncio.apply()

import logging

from llama_index.core import DocumentSummaryIndex, SimpleDirectoryReader,StorageContext, load_index_from_storage, Settings, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=3600.0)


print(os.curdir)
documents = SimpleDirectoryReader(input_files=["./src/Data/lambda.pdf"],filename_as_id=True).load_data()
splitter = SentenceSplitter(chunk_size=1024)

persist_directory = './Summary_Index'
if(not os.path.exists(persist_directory)):
    print("Indexing started.")
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize",use_async=True)
    index = DocumentSummaryIndex.from_documents(documents,transformations=[splitter],response_synthesizer = response_synthesizer, show_progress= True)
    index.storage_context.persist(persist_dir=persist_directory)
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    index = load_index_from_storage(storage_context)

print(index.ref_doc_info)

try:
    if len(sys.argv) > 0 :
        paths = sys.argv[1:]
        for path in paths:
            print(path)
            document = SimpleDirectoryReader(input_files=[path]).load_data()
            for page in document:
                index.insert(page)
except Exception as e:
    print('document insertion failed',e)

chat_engine = index.as_chat_engine()

# chatting logic
user_input = ""
while(user_input != "bye"):
    user_input = input("you: ")
    bot_output = chat_engine.chat(user_input)
    print(f"bot: {bot_output}")