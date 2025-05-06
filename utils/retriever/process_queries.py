import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import yake
import spacy


OUTPUT_FILE = "./IR/results/clean_query.json"

nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("pt_core_news_md")

def is_related_to_legal_term(word, target_words=None, threshold=0.6):
    if target_words is None:
        target_words = ["lei"]

    target_synsets = []
    for term in target_words:
        target_synsets.extend(wordnet.synsets(term, pos='n', lang='por'))

    word_synsets = wordnet.synsets(word, pos='n', lang='por')

    for ws in word_synsets:
        for ts in target_synsets:
            sim = ws.wup_similarity(ts)
            if sim and sim > threshold:
                return True
    return False

def preprocess_query(query, use_yake=True):
    exception_words = {"menores", "menor"}

    if use_yake:
        length = len(query.split())
        _dedupL = 0.5 if 10 < length < 30 else 0.8 if length >= 30 else 0.3

        # Use n=1 to prioritize single words
        kw_extractor = yake.KeywordExtractor(lan="pt", n=1, dedupLim=_dedupL, dedupFunc="seqm")
        keywords = [kw[0] for kw in kw_extractor.extract_keywords(query)]

        text_to_process = " ".join(keywords)
    else:
        text_to_process = query

    print(f"[IR] preprocess_1: {text_to_process}")
    # Process text with spaCy
    doc = nlp(text_to_process)
    
    candidate_keywords = []
    for token in doc:
        word = token.text.lower()
        if word in exception_words:
            candidate_keywords.append(word)
        elif token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ORG", "GPE" } and not token.is_stop and token.is_alpha:
            candidate_keywords.append(token.lemma_.lower())

    print(f"[IR] preprocess_2: {candidate_keywords}")
    # Filter core content words
    final_keywords = [
        kw for kw in candidate_keywords if not is_related_to_legal_term(kw) and len(kw) > 2
    ]

    print(f"[IR] preprocess_3: {list(set(final_keywords))}")
    
    return list(set(final_keywords))
