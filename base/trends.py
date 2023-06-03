#Linear algebra and data manipulation
import pandas as pd
import numpy as np

#Performance Metrics
from sklearn.metrics import silhouette_score, pairwise_distances

#NLP
from sentence_transformers import SentenceTransformer

#Preprocessing
from .ctfidf import CTFIDFVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Algorithms
import umap
import hdbscan

from .news_scaping import build_data

def get_doc_class_datafram(data_column, clusters):
    docs = []

    for i, doc in enumerate(data_column):
        docs.append([doc, clusters.labels_[i]])

    docs = pd.DataFrame(docs, columns=['document', 'class'])    
    docs.dropna(inplace=True)

    return docs

def get_important_words_per_class(docs_per_class, docs):
    count_vectorizer = CountVectorizer().fit(docs_per_class['document'])
    count = count_vectorizer.transform(docs_per_class['document'])
    words = count_vectorizer.get_feature_names()
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()
    words_per_class = {label: [words[index] for index in ctfidf[label].argsort()[-5:]] for label in docs_per_class['class']}
    important_words_per_class = {}

    for k, v in words_per_class.items():
        words_per_class[k] = v[::-1] 

    for topic, entities in words_per_class.items():
        words = []

        for entity in entities:
            is_unique = True

            for curr_topic, curr_entities in words_per_class.items():
                if curr_topic != topic:
                    if entity in curr_entities:
                        is_unique = False
                        break          
            if is_unique:
                words.append(entity)

        important_words_per_class[topic] = words 

    return important_words_per_class



def get_topic_percentage(data_column, important_words_per_class, documents_without_random_class_count):
    topic_percentage = {}
    bow = CountVectorizer(binary=True)
    X_bow = bow.fit_transform(data_column.values).toarray()
    df = pd.DataFrame(X_bow, columns=bow.get_feature_names())

    for topic, entities in important_words_per_class.items():
        occurance_percentage = []

        for entity in entities:
            occurance_percentage.append(np.sum(df[entity]) / documents_without_random_class_count)

        topic_percentage[topic] = np.max(occurance_percentage)

    return topic_percentage


def get_trends():
    column_name = 'cleaned_interested_content'
    distilbert_model_512 = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    trends = []
    data = build_data()
    data.dropna(inplace=True)

    embedding_model = distilbert_model_512
    X_BERT_vec_512 = distilbert_model_512.encode(list(data[column_name]))
    X_BERT_vec_5 = umap.UMAP(n_components=5, n_neighbors=15, metric='cosine', random_state=0).fit_transform(X_BERT_vec_512)

    distance = pairwise_distances(X_BERT_vec_5, metric='cosine')
    clusters = hdbscan.HDBSCAN(gen_min_span_tree=True, metric='precomputed')
    clusters.fit(distance.astype('float64'))

    docs = get_doc_class_datafram(data[column_name], clusters) 
    docs_per_class = docs.groupby(['class'], as_index=False).agg({'document': ' '.join})
    important_words_per_class = get_important_words_per_class(docs_per_class, docs)


    documents_without_random_class_count = docs.shape[0] - docs['class'].value_counts()[-1]

    topic_percentage = get_topic_percentage(data[column_name], 
                                            important_words_per_class,
                                            documents_without_random_class_count)

    topic_percentage_list = [val for key, val in topic_percentage.items() if key != -1]
    thershold = np.median(topic_percentage_list)
    
    for topic, percentage in topic_percentage.items():
        if percentage >= thershold and topic != -1:
            trends.append(important_words_per_class[topic])

    return trends