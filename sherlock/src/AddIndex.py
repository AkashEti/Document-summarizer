from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage

def add_index(path):
    document = SimpleDirectoryReader(input_files=[path])
    persist_directory = './Summary_Index'
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    index = load_index_from_storage(storage_context)
    index.insert(document)