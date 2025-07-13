from pymongo import MongoClient
from datetime import datetime, timezone
import time
from utils.logger import api_logger

class MongoUtils:
    def __init__(self, uri="mongodb://localhost:27017", db_name="rag_db"):
        self.client = MongoClient(uri)               # Connect to MongoDB
        self.db = self.client[db_name]               # Select the database
        self.docs_collection = self.db["uploaded_documents"]
        self.chat_collection = self.db["chat_sessions"]

    def save_uploaded_document(self, user_id, filename, file_content=None):
        start_time = time.time()
        
        try:
            api_logger.info(f"DB: Saving uploaded document | User: {user_id} | File: {filename}")
            
            doc = {
                "user_id": user_id,
                "filename": filename,
                "file_content": file_content,  # Store actual file content
                "upload_time": datetime.now(timezone.utc)
            }
            
            result = self.docs_collection.insert_one(doc)
            
            save_time = time.time() - start_time
            api_logger.info(f"DB: Document saved successfully in {save_time:.3f}s | User: {user_id} | File: {filename} | Doc ID: {result.inserted_id}")
            
        except Exception as e:
            save_time = time.time() - start_time
            api_logger.error(f"DB: Failed to save document in {save_time:.3f}s | User: {user_id} | File: {filename}", exc_info=True)
            raise

    def get_documents_for_user(self, user_id):
        start_time = time.time()
        
        try:
            api_logger.info(f"DB: Retrieving documents for user: {user_id}")
            
            documents = list(self.docs_collection.find({"user_id": user_id}))
            
            retrieve_time = time.time() - start_time
            api_logger.info(f"DB: Documents retrieved successfully in {retrieve_time:.3f}s | User: {user_id} | Count: {len(documents)}")
            
            return documents
            
        except Exception as e:
            retrieve_time = time.time() - start_time
            api_logger.error(f"DB: Failed to retrieve documents in {retrieve_time:.3f}s | User: {user_id}", exc_info=True)
            raise

    def save_chat_session(self, user_id, session_id, query, response):
        start_time = time.time()
        
        try:
            api_logger.info(f"DB: Saving chat session | User: {user_id} | Session: {session_id}")
            
            doc = {
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
                "response": response,
                "timestamp": datetime.now(timezone.utc)
            }
            
            result = self.chat_collection.insert_one(doc)
            
            save_time = time.time() - start_time
            api_logger.info(f"DB: Chat session saved successfully in {save_time:.3f}s | User: {user_id} | Session: {session_id} | Doc ID: {result.inserted_id}")
            
            return user_id
            
        except Exception as e:
            save_time = time.time() - start_time
            api_logger.error(f"DB: Failed to save chat session in {save_time:.3f}s | User: {user_id} | Session: {session_id}", exc_info=True)
            raise

    def get_chat_history(self, session_id):
        """Get last 3 chats in chronological order (3rd last, 2nd last, last)."""
        start_time = time.time()
        
        try:
            api_logger.info(f"DB: Retrieving chat history | Session: {session_id}")
            
            # Get last 3 chats (latest first)
            chats = list(self.chat_collection.find({"session_id": session_id}).sort("timestamp", -1).limit(3))
            
            # Reverse to get chronological order (3rd last first, then 2nd last, then last)
            chats.reverse()
            
            # Format chat history as requested
            formatted_history = []
            for chat in chats:
                formatted_history.append(f"QUERY: {chat.get('query', '')}")
                formatted_history.append(f"RESPONSE: {chat.get('response', '')}")
            
            retrieve_time = time.time() - start_time
            api_logger.info(f"DB: Chat history retrieved successfully in {retrieve_time:.3f}s | Session: {session_id} | Chats: {len(chats)}")
            
            return formatted_history
            
        except Exception as e:
            retrieve_time = time.time() - start_time
            api_logger.error(f"DB: Failed to retrieve chat history in {retrieve_time:.3f}s | Session: {session_id}", exc_info=True)
            raise
        

    def delete_chat_session(self, session_id):
        self.chat_collection.delete_many({"session_id": session_id})
