from sklearn.metrics import silhouette_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import json

IMG_PATH = './src/static/img/'

plt.switch_backend('agg')
device = 0
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length = 128)
# model = AutoModel.from_pretrained("bert-base-uncased").to(device)

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic

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
        embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        # umap_model = DimensionalityReduction()
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', init = 'pca')
        hdbscan_model = AgglomerativeClustering(n_clusters=n_topics)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)# True
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

    def __init__(self, df, n_topics_list):
        """
        初始化 ClusteringWithTopic，接受一个 n_topics_list，其中包含多个聚类数目，
        选取 silhouette_score 最高的结果。
        """
        embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        self.embeddings = embedding_model.encode(df, show_progress_bar=True)
        
        self.df = df
        self.n_topics_list = n_topics_list

        self.embedding_model = embedding_model
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',init ='pca')
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)
        self.keybert_model = KeyBERTInspired()
        self.representation_model = {"KeyBERT": self.keybert_model}
        
        # 用于存储不同聚类数目的结果
        self.best_n_topics = None
        self.best_labels = None
        self.best_score = -1 
    # def fit_and_get_labels(self, X):
    #     topics, probs = self.topic_model.fit_transform(self.df, self.embeddings)
    #     return topics 
    def fit_and_get_labels(self):
        """
        对不同的 n_topics 进行聚类，计算 silhouette_score，选取最佳的 n_topics 进行后续操作。
        """
        for n_topics in self.n_topics_list:
            hdbscan_model = AgglomerativeClustering(n_clusters=n_topics)

            topic_model = BERTopic(
                embedding_model= self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                representation_model=self.representation_model,
                top_n_words=10,
                verbose=False
            )

            topics, _ = topic_model.fit_transform(self.df, self.embeddings)
            
            # 计算 silhouette_score
            if len(set(topics)) > 1:  # silhouette_score 需要至少 2 个类别
                score = silhouette_score(self.embeddings, topics)
            else:
                score = -1  # 单个类别时，silhouette_score 无意义

            print(f"n_topics={n_topics}, silhouette_score={score}")

            # 记录最佳的 n_topics
            if score > self.best_score:
                self.best_score = score
                self.best_n_topics = n_topics
                self.best_labels = topics
                self.best_topic_model = topic_model
        
        print(f"Best n_topics={self.best_n_topics}, Best silhouette_score={self.best_score}")
        return self.best_labels, self.best_topic_model, self.best_n_topics

def clustering(df, n_cluster, survey_id):
    text = df['retrieval_result'].astype(str)
    clustering = ClusteringWithTopic(text, n_cluster)
    df['label'] = clustering.fit_and_get_labels(text)
    
    print("The clustering result is: ")
    for col in df.columns:
        print(f"{col}: {df.iloc[0][col]}")
    
    # Save topic model information as JSON
    topic_json = clustering.topic_model.get_topic_info().to_json()
    with open(f'./src/static/data/info/{survey_id}/topic.json', 'w', encoding="utf-8") as file:
        file.write(topic_json)
    
    # Create a dictionary from 'ref_title' and 'retrieval_result' columns
    description_dict = dict(zip(df['ref_title'], df['retrieval_result']))
    
    # Save the dictionary to description.json
    with open(f'./src/static/data/info/{survey_id}/description.json', 'w', encoding="utf-8") as file:
        json.dump(description_dict, file, ensure_ascii=False, indent=4)
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
    output_tsv_filename = "./src/static/data/tsv/" + survey_id + '.tsv'
    df.to_csv(output_tsv_filename, sep='\t')
    return df, colors

def clustering(df, n_topics_list, survey_id, info_path='./src/static/data/info', tsv_path='./src/static/data/tsv'):
    text = df['retrieval_result'].astype(str)
    clustering = ClusteringWithTopic(text, n_topics_list)
    df['label'], topic_model, best_n_topics = clustering.fit_and_get_labels()

    print("The clustering result is: ")
    for col in df.columns:
        print(f"{col}: {df.iloc[0][col]}")

    # 保存 topic model 信息
    topic_json = topic_model.get_topic_info().to_json()
    with open(f'{info_path}/{survey_id}/topic.json', 'w', encoding="utf-8") as file:
        file.write(topic_json)

    # 创建描述信息
    description_dict = dict(zip(df['ref_title'], df['retrieval_result']))
    with open(f'{info_path}/{survey_id}/description.json', 'w', encoding="utf-8") as file:
        json.dump(description_dict, file, ensure_ascii=False, indent=4)

    # t-SNE 降维可视化
    X = np.array(clustering.embeddings)
    perplexity = min(10, max(1, X.shape[0] // 2))  # 避免 perplexity 过大

    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    colors = scatter(X_tsne, df['label'])  # 计算颜色

    # plt.savefig(IMG_PATH + 'tsne_' + survey_id + '.png', dpi=800, transparent=True)

    # plt.close()
    output_tsv_filename = f"{tsv_path}/{survey_id}.tsv"
    df.to_csv(output_tsv_filename, sep='\t')
    return df, colors, best_n_topics

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
