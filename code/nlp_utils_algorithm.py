import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import lexicons
import re
from itertools import combinations
import numpy as np
from gensim.models import KeyedVectors
import networkx as nx

model = KeyedVectors.load('data/model.model')

def prune_text(text):
    mwe_tokenizer = MWETokenizer(lexicons.mwe_list)
    tokens = word_tokenize(text)
    tokens = mwe_tokenizer.tokenize(tokens)
    tokens = [token.lower() for token in tokens]
    tokens = lemmatize_tokens(tokens)
    tokens = [token for token in tokens if is_token_in_model(token)]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    tokens = mwe_tokenizer.tokenize(tokens)

    sentences = []
    sentence = []
    for token in tokens:
        sentence.append(token)
        if token in ['.', '?', '!']:
            sentence = [t for t in sentence if re.match(r'^[a-zA-Z0-9_-]+$', t)]
            sentences.append(sentence)
            sentence = []
    tokens = [token for token in tokens if re.match(r'^[a-zA-Z0-9_-]+$', token)]
    return tokens, sentences

def lemmatize_tokens(tokens):
    tag_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, tag_map.get(pos[0], wordnet.NOUN)) for token, pos in pos_tags if pos[0] in ['N', 'J', 'V', 'R', '.']]

def is_token_in_model(token):
    tokens = token.split("_")
    for word in tokens:
        if word not in model:
            return False
    return True

def get_co(sentences, representation='dictionary', window_size=3):
    unique_tokens = list(set([token for sentence in sentences for token in sentence]))
    n = len(unique_tokens)
    index_dict = {key: index for index, key in enumerate(unique_tokens)} 
    co = {} if representation == 'dictionary' else np.zeros((n,n))

    def populate(co, window_tokens):
        pairs = list(combinations(window_tokens, 2))
        for pair in pairs:
            if representation == 'dictionary':
                pair = tuple(sorted(pair))
                if pair not in co:
                    co[pair] = 0
                co[pair] += 1
            elif representation == 'matrix':
                t1, t2 = pair
                co[index_dict[t1]][index_dict[t2]] += 1
                co[index_dict[t2]][index_dict[t1]] += 1
        return co  

    for sentence in sentences:
        short_sentence = True
        for i, _ in enumerate(sentence):
            if i + window_size > len(sentence):
                break
            short_sentence = False
            window_tokens = sentence[i:i+window_size]
            co = populate(co, window_tokens)
        if short_sentence:
            co = populate(co, sentence)
    return co, index_dict

def get_word_em(token):
    tokens = token.split("_")
    embeddings = []
    for word in tokens:
        if word in model:
            embeddings.append(model[word])
        else:
            raise Exception(f"Word {token} not present in vocabulary!")
    return sum(embeddings)/len(embeddings)

def personalized_pagerank(graph_dict, seeds, alpha=0.85):
    G = nx.Graph()
    for (u, v), weight in graph_dict.items():
        G.add_edge(u, v, weight=weight)
    personalization = {node: 0 for node in G.nodes()}
    for seed in seeds:
        if seed in personalization:
            personalization[seed] = 1.0
    total = sum(personalization.values())
    if total > 0:
        personalization = {k: v / total for k, v in personalization.items()}
    return nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')

if __name__ == "__main__":
    text = "The island country of Japan has developed into a great economy after World War 2. The Japan sea is a source of fish. Sushi is a famous fish and rice food."
    tokens, sentences = prune_text(text)
    print("Tokens:", tokens)
    print("Sentences:", sentences)
    co_dict, _ = get_co(sentences)
    print("Co-occurrence Dictionary:", co_dict)
    seeds = ['japan', 'fish']
    pr_scores = personalized_pagerank(co_dict, seeds)
    print("Personalized PageRank:", pr_scores)
