import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, collection_name="questions", data_path="data/questions.json"):
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.data_path = Path(data_path)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedder,
            persist_directory="data/chroma"
        )
        self.load_questions()

    def load_questions(self):
        """Load questions from JSON and store in ChromaDB."""
        if self.vector_store._collection.count() == 0:
            with open(self.data_path, "r") as f:
                questions = json.load(f)
            
            documents = [
                Document(
                    page_content=q["question"],
                    metadata={
                        "id": q["id"],
                        "topic": q["topic"],
                        "answer_key": q["answer_key"],
                        "difficulty": q["difficulty"]
                    }
                ) for q in questions
            ]
            self.vector_store.add_documents(documents)

    def retrieve_question(self, topic, difficulty, query=""):
        """Retrieve a question by topic and difficulty."""
        if not query:
            query = f"{topic} {difficulty} interview question"
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=1,
            filter={"$and": [
                {"topic": {"$eq": topic}},
                {"difficulty": {"$eq": difficulty}}
            ]}
        )
        if results:
            doc, _ = results[0]
            return {
                "question": doc.page_content,
                "answer_key": doc.metadata["answer_key"],
                "difficulty": doc.metadata["difficulty"]
            }
        return None
