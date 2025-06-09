from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from dotenv import load_dotenv
import requests
import spacy
import nltk
import yake
import os

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

    #print(f"[IR] preprocess_1: {text_to_process}")
    # Process text with spaCy
    doc = nlp(text_to_process)
    
    candidate_keywords = []
    for token in doc:
        word = token.text.lower()
        if word in exception_words:
            candidate_keywords.append(word)
        elif token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ORG", "GPE" } and not token.is_stop and token.is_alpha:
            candidate_keywords.append(token.lemma_.lower())

    #print(f"[IR] preprocess_2: {candidate_keywords}")
    # Filter core content words
    final_keywords = [
        kw for kw in candidate_keywords if not is_related_to_legal_term(kw) and len(kw) > 2
    ]

    #print(f"[IR] preprocess_3: {list(set(final_keywords))}")
    
    return list(set(final_keywords))


def process_user_query(user_query, llm_communicator):
    system_prompt = f"""
Estás encarregado de interpretar perguntas de utilizadores relativas a documentos legislativos portugueses.

Recebes perguntas em linguagem natural e deves extrair delas:
- a intenção principal de pesquisa ("query")
- quaisquer filtros estruturados com base nos campos dos documentos
- uma explicação ("reasoning") da tua interpretação

### Campos disponíveis nos documentos:
- ID: código da lei (ex: "Lei n.º 4/85")
- Titulo: título completo
- Sumario: resumo da lei
- Data_ultima_modificacao: data da última modificação no formato YYYY-MM-DD
- Modificacao: referência do Diário da República
- Url: link para o documento

### Objetivo:
Converter perguntas em JSON estruturado com os seguintes campos:
- "query": intenção principal (ex: "leis sobre remuneração de cargos políticos")
- "filters": dicionário com chaves como "Data_ultima_modificacao", "Titulo", "ID", etc.
- "reasoning": explicação do teu raciocínio

### Exemplos:

Input: "Mostra-me todas as leis sobre divórcio e guarda de menores."
Output:
{{
  "query": "leis sobre divórcio e guarda de menores",
  "filters": {{
    "$match": {{
      "$text": {{
        "$search": "\\"divórcio\\" \\"guarda de menores\\""
      }}
    }}
  }},
  "reasoning": "A pergunta indica interesse por leis sobre divórcio e leis sobre guarda de menores."
}}

Input: "Decreto-Lei 44 2011, Lei n.º 82-D/2014"
Output:
{{
  "query": "leis específicas mencionadas",
  "filters": {{
    "$match": {{
      "$or": [
        {{
          "$and": [
            {{ "ID": {{ "$regex": "Decreto-Lei", "$options": "i" }} }},
            {{ "ID": {{ "$regex": "44", "$options": "i" }} }},
            {{ "ID": {{ "$regex": "2011", "$options": "i" }} }}
          ]
        }},
        {{
          "$and": [
            {{ "ID": {{ "$regex": "Lei", "$options": "i" }} }},
            {{ "ID": {{ "$regex": "82", "$options": "i" }} }},
            {{ "ID": {{ "$regex": "D", "$options": "s" }} }},
            {{ "ID": {{ "$regex": "2014", "$options": "i" }} }}
          ]
        }}
      ]
    }}
  }},
  "reasoning": "A pergunta indica interesse por leis específicas mencionadas pelo utilizador."
}}

## Contexto:
Input: "{user_query}"
Output:
"""

    response = llm_communicator.generate(
        prompt=system_prompt,
        temperature=0.0,
        top_p=0.9
    )

    if response is not None:
        return response
    else:
        return "Error: LLM request failed."