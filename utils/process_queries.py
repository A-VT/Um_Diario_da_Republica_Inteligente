import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import yake
import spacy
import os
import json


OUTPUT_FILE = "./IR/results/clean_query.json"

nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("pt_core_news_md")

def save_results(results):
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

def preprocess_query(query):
    #define keywords
    custom_kw_extractor = yake.KeywordExtractor(lan="pt", n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=5, features=None)
    keywords = list(custom_kw_extractor.extract_keywords(query))
    just_keyw = [i[0] for i in keywords]

    #remove words of lexical field 'legislação'
    target_words = ["legislação", "leis", "normas", "decretos-lei"]
    target_synsets = []
    for word in target_words:
        target_synsets.extend(wordnet.synsets(word, pos='n', lang="por"))

    unrelated_keywords = []
    for kw in just_keyw:
        word_synsets = wordnet.synsets(kw, pos='n', lang="por")
        is_related = False
        for word_syn in word_synsets:
            for target_syn in target_synsets:
                similarity = word_syn.wup_similarity(target_syn)
                if similarity and similarity > 0.6:
                    is_related = True
                    break
            if is_related:
                break
        if not is_related:
            unrelated_keywords.append(kw)
    
    #convert unrelated keywords to lowercase
    unrelated_keywords = [word.lower() for word in unrelated_keywords]
    
    #keep nouns and verbs, and lemmatize
    #doc = nlp(" ".join(unrelated_keywords))
    #filtered_keywords = [
    #    token.lemma_
    #    for token in doc
    #    if token.pos_ in ["NOUN", "VERB"]
    #]

    # Remove duplicates and return final keywords
    final_keywords = list(set(unrelated_keywords))
    save_results(final_keywords)

    return final_keywords