import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from summa import keywords
TaggededDocument = gensim.models.doc2vec.TaggedDocument

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from operator import itemgetter
import traceback


import nltk
from nltk.tokenize import word_tokenize,sent_tokenize


import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from sklearn.manifold import TSNE


import nltk
from fuzzywuzzy import fuzz

from summa.summarizer import summarize
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
import torch
from sklearn.cluster import AgglomerativeClustering
nlp = spacy.load("en_core_sci_sm")

IMG_PATH = './src/static/img/'

plt.switch_backend('agg')
device = 0
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length = 128)
# model = AutoModel.from_pretrained("bert-base-uncased").to(device)

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from bertopic import BERTopic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class DimensionalityReduction:
    def fit(self, X):
        return self

    def transform(self, X):
        return X

class ClusteringWithTopic:
    def __init__(self, df, n_topics=3):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = DimensionalityReduction()
        hdbscan_model = AgglomerativeClustering(n_clusters=n_topics)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        keybert_model = KeyBERTInspired()

        self.df = df
        self.embeddings = embeddings = embedding_model.encode(df, show_progress_bar=True)

        representation_model = {
        "KeyBERT": keybert_model,
        # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        # "MMR": mmr_model,
        # "POS": pos_model
    }
        self.topic_model = BERTopic(

        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,

        # Hyperparameters
        top_n_words=10,
        verbose=True
        )
    def fit_and_get_labels(self, X):
        topics, probs = self.topic_model.fit_transform(self.df, self.embeddings)
        return topics

def clustering(df, n_cluster, survey_id):
    text = df['retrieval_result'].astype(str)
    clustering = ClusteringWithTopic(text, n_cluster)
    df['label'] = clustering.fit_and_get_labels(text)
    print(clustering.topic_model.get_topic_info())
    topic_json = clustering.topic_model.get_topic_info().to_json()
    with open(f'./src/static/data/info/{survey_id}/topic.json', 'w') as file:
        file.write(topic_json)
    # df['top_n_words'] = clustering.topic_model.get_topic_info()['Representation'].tolist()
    # df['topic_word'] = clustering.topic_model.get_topic_info()['KeyBERT'].tolist()


    X = np.array(clustering.embeddings)
    perplexity = 10
    if X.shape[0] <= perplexity:
        perplexity = max(1, X.shape[0] // 2)   

    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    colors = scatter(X_tsne, df['label'])

    plt.savefig(IMG_PATH + 'tsne_' + survey_id + '.png', dpi=800, transparent=True)

    plt.close()
    return df, colors

def scatter(x, colors):
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # We choose a color palette with seaborn.
    palette = np.array(sns.hls_palette(8, l=0.4, s=.8))
    color_hex = sns.color_palette(sns.hls_palette(8, l=0.4, s=.8)).as_hex()
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=1,
                    c=palette[colors.astype(np.int32)])
    c = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in colors]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], '[' + str(i) + ']', fontsize=20, color=c[i], weight='1000')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    return color_hex[:colors.nunique()]
