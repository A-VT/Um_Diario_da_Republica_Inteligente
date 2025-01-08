from pymongo import MongoClient
#from dotenv import load_dotenv
#import os

def connect_to_mongo():
    #load_dotenv()
    mongo_user = os.getenv('MONGO_USER')
    mongo_password = os.getenv('MONGO_PASSWORD')
    try:
        mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@drbd.bbw8o.mongodb.net/?retryWrites=true&w=majority&appName=DRbd"
        client = MongoClient(mongo_uri)
        db = client['DiarioRepublica']
        collection_dados = db['dados']
        collection_metadados = db['metadados']
        print("Connected to MongoDB.")
        return client, db, collection_dados, collection_metadados
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None, None, None
    

def connect_to_mongo(mongo_user, mongo_password):
    try:
        mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@drbd.bbw8o.mongodb.net/?retryWrites=true&w=majority&appName=DRbd"
        client = MongoClient(mongo_uri)
        db = client['DiarioRepublica']
        collection_dados = db['dados']
        collection_metadados = db['metadados']
        print("Connected to MongoDB.")
        return client, db, collection_dados, collection_metadados
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None, None, None