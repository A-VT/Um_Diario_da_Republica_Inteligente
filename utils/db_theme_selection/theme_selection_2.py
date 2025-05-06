import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from dotenv import load_dotenv
from utils.mongo_conn import connect_to_mongo


# Load spaCy model for Portuguese
nlp = spacy.load("pt_core_news_md")

# Load stopwords in Portuguese
stop_words = stopwords.words("portuguese")

class LegalDocumentTopicModeling:
    def __init__(self):
        self.client, self.db, self.collection_dados, self.collection_metadados = self._connect_to_db()
    
    def _connect_to_db(self):
        load_dotenv()
        cred_mongo_user, cred_mongo_password = os.getenv("MONGO_USER"), os.getenv("MONGO_PASSWORD")
        return connect_to_mongo(cred_mongo_user, cred_mongo_password)
    
    # Fetch documents from MongoDB
    def fetch_documents(self):
        query = {"Titulo": {"$exists": True}, "Sumario": {"$exists": True}}
        projection = {"Titulo": 1, "Sumario": 1}
        documents = list(self.collection_metadados.find(query, projection))
        return pd.DataFrame(documents)

    # Preprocess the documents: lowercase, remove punctuation, and stopwords
    def preprocess_text(self, text):
        text = text.lower()
        doc = nlp(text)
        tokens = [token.text for token in doc if token.text not in stop_words and not token.is_stop]
        return " ".join(tokens)

    # Apply topic modeling (LDA or NMF)
    def topic_modeling(self, documents, n_topics=5, model_type='LDA'):
        # Preprocess the documents (titles + summaries)
        documents_cleaned = [self.preprocess_text(doc) for doc in documents]
        
        # Vectorize the documents using TF-IDF
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))  # bigrams
        X = vectorizer.fit_transform(documents_cleaned)
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply the specified topic modeling method (LDA or NMF)
        if model_type == 'LDA':
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            model.fit(X)
            topics = model.components_
        elif model_type == 'NMF':
            model = NMF(n_components=n_topics, random_state=42)
            model.fit(X)
            topics = model.components_
        elif model_type == 'KMeans':
            model = KMeans(n_clusters=n_topics, random_state=42)
            model.fit(X)
            topics = model.cluster_centers_
        else:
            raise ValueError("model_type must be 'LDA', 'NMF', or 'KMeans'")

        model.fit(X)
        
        # Display the topics
        output_lines = []
        for topic_idx, topic in enumerate(topics):
            top_terms = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topic_str = f"Topic {topic_idx + 1}: " + " ".join(top_terms)
            output_lines.append(topic_str)

        os.makedirs("results", exist_ok=True)
        output_file = f"./utils/db_theme_selection/results/{model_type}_{n_topics}topics.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nTopics saved to {output_file}")


    def run(self, n_topics=5, model_type='LDA'):
        self._connect_to_db()
        
        documents_df = self.fetch_documents()
        
        documents = documents_df["Titulo"].fillna('') + " " + documents_df["Sumario"].fillna('')
        
        self.topic_modeling(documents, n_topics, model_type)


# Example usage
if __name__ == "__main__":
    legal_document_modeling = LegalDocumentTopicModeling()
    legal_document_modeling.run(n_topics=20, model_type='NMF')  # You can use 'LDA', 'NMF' or "KMeans" here
