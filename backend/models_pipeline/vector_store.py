import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from models_pipeline.input_processor_chunking_audio import ProcessInput
import os
# import textract
from transformers import WhisperForConditionalGeneration, WhisperProcessor


model_path = "./models_pipeline/clients/whisper_model_small"
processor_path = "./models_pipeline/clients/whisper_processor_small"

whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
whisper_processor = WhisperProcessor.from_pretrained(processor_path)

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collections = {}
    
    def test_chroma_storage(self):
        print("Testing Chroma storage...")
        test_collection_name = "test_collection"
        
        test_collection_2 = self.chroma_client.get_or_create_collection(name=test_collection_name)
        print(f"Collection '{test_collection_name}' accessed.")

        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        test_id = "test_id"
        print("Adding test embedding...")
        test_collection_2.add( 
            embeddings=[test_embedding],
            documents=["test document"],
            metadatas=[{"test": "metadata"}],
            ids=[test_id]
        )
 
        print("Retrieving test embedding...")
        test_result = test_collection_2.get(ids=[test_id])
        print(f"Retrieved data: {test_result}")

        if 'embeddings' in test_result and test_result['embeddings'] is not None:
            print(f"Test embedding storage: {'Successful' if test_result['embeddings'][0] == test_embedding else 'Failed'}")
            print(f"Retrieved embeddings: {test_result['embeddings']}")
        else:
            print("Error: Embeddings not found in retrieved data")
    
        print("Collection peek:")
        print(test_collection_2.peek())
        # print(f"Test embedding storage: {'Successful' if test_result['embeddings'][0] == test_embedding else 'Failed'}")
        # print(f"Retrieved embeddings: {test_result['embeddings']}")


    def get_or_create_collection(self, topic):
        collection_name = f"{topic}"
        if collection_name not in self.collections:
            self.collections[collection_name] = self.chroma_client.get_or_create_collection(name=collection_name)
        print("get_or_create")
        return self.collections[collection_name]

    def add_pdf_file_to_collection(self, file_path, topic):
        processor = ProcessInput(whisper_model,whisper_processor)
        collection = self.get_or_create_collection(topic)
        text = processor.process_pdf(file_path)
        print(text)
        embeddings = self.embedding_model.encode(text).tolist()

        
        file_name = os.path.basename(file_path)
        
        # Check if the file already exists in the collection
        existing_ids = collection.get(ids=[file_name])['ids']
        
        if existing_ids:
            # Update the existing entry
            collection.update(
                embeddings=[embeddings],
                documents=[text],
                metadatas=[{"file_name": file_name, "topic": topic}],
                ids=[file_name]
            )
            print('AIGHT CLOWJN')
        else:
            # Add a new entry
            collection.add(
                embeddings=[embeddings],
                documents=[text],
                metadatas=[{"file_name": file_name, "topic": topic}],
                ids=[file_name]
            )
            print('TFFFF')
    def add_pdf_text_file_to_collection(self, file_name, text, topic):
        collection = self.get_or_create_collection(topic)
        # text = self.process_pdf(file_path)
        embeddings = self.embedding_model.encode(text).tolist()
        # file_name = os.path.basename(file_path)
        print("----------------------------")
        print(f"Generated embeddings: {embeddings[:5]}...")
        print(f"Embeddings type: {type(embeddings)}, Length: {len(embeddings)}")
        
        # Check if the file already exists in the collection
        existing_ids = collection.get(ids=[f"{topic}"])['ids']

        print("Adding to collection...")
        try:
            if existing_ids:
            # Update the existing entry
                collection.update(
                    embeddings=[embeddings],
                    documents=[text],
                    metadatas=[{"file_name": file_name, "topic": topic}],
                    ids=[f"{topic}"]
                )
            else:
                # Add a new entry
                collection.add(
                    embeddings=[embeddings],
                    documents=[text],
                    metadatas=[{"file_name": file_name, "topic": topic}],
                    ids=[f"{topic}"]
                )
                print(f"Added document {file_name} to collection. Embeddings stored: {embeddings[:5]}...")
        except Exception as e:
            print(f"Error adding to collection: {str(e)}")
            return
        
        print("Verifying addition...")
        added_data = collection.get(ids=[f"{topic}"])
        print(f"Verified data - Embeddings: {'None' if added_data['embeddings'] is None else 'Not None'}")
        if added_data['embeddings'] is not None:
            print(f"First 5 elements of stored embedding: {added_data['embeddings'][0][:5]}")
            
        # if existing_ids:
        #     # Update the existing entry
        #     collection.update(
        #         embeddings=[embeddings],
        #         documents=[text],
        #         metadatas=[{"file_name": file_name, "topic": topic, "field": field}],
        #         ids=[file_name]
        #     )
        # else:
        #     # Add a new entry
        #     collection.add(
        #         embeddings=[embeddings],
        #         documents=[text],
        #         metadatas=[{"file_name": file_name, "topic": topic, "field": field}],
        #         ids=[file_name]
        #     )
        #     print(f"Added document {file_name} to collection. Embeddings stored: {embeddings[:5]}...")
    

    # def add_student_data(self):
    #     collection_name = 

 
    def inspect_collection(self,  collection_name):
        if collection_name in self.collections:
            collection = self.collections[collection_name]
            print(f"Collection {collection_name} contents:")
            all_data = collection.get()
            print(f"IDs: {all_data['ids']}")
            print(f"Metadata: {all_data['metadatas']}")
            # print(f"Embeddings: {[emb[:5] for emb in all_data['embeddings']]}")
            if all_data['embeddings'] is None:
                print("Embeddings: None")
            else:
                print(f"Embeddings: {[emb[:5] if emb else None for emb in all_data['embeddings']]}")
        else:
            print(f"Collection {collection_name} not found")


    def search_context(self, query, topic=None, field=None, n_results=1):
        print ("beginning search")
        query_embeddings = self.embedding_model.encode(query).tolist()
        if topic and field:
            print("if topic and field")
            collection_name = f"{topic}-{field}"
            if collection_name in self.collections:
                results= self.collections[collection_name].query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    include=['metadatas', 'documents', 'distances']
                )
                print(f"Search results: {results}")
                return results
            else:
                print(f"Collection {collection_name} not found")
                return []
        else:
            print("NOT if topic and field")
            results = []
            for collection in self.collections.values():
                results.extend(collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results
                ))
            return results[:n_results]
        
    # def search(self, topic=None, field=None, n_results=1):
    #     print ("beginning search")
    #     # query_embeddings = self.embedding_model.encode(query).tolist()
    #     if topic and field:
    #         print("if topic and field")
    #         collection_name = f"{topic}-{field}"
    #         if collection_name in self.collections:
    #             results= self.collections[collection_name].query(
    #                 query_embeddings=query_embeddings,
    #                 n_results=n_results,
    #                 include=['metadatas', 'documents', 'distances']
    #             )
    #             print(f"Search results: {results}")
    #             return results
    #         else:
    #             print(f"Collection {collection_name} not found")
    #             return []
    #     else:
    #         print("NOT if topic and field")
    #         results = []
    #         for collection in self.collections.values():
    #             results.extend(collection.query(
    #                 query_embeddings=query_embeddings,
    #                 n_results=n_results
    #             ))
    #         return results[:n_results]

    def list_collections(self):
        return list(self.collections.keys())

    def get_collection_info(self, collection_name):
        if collection_name in self.collections:
            collection = self.collections[collection_name]
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.get()
            }
        else:
            return None