from django.shortcuts import render
from .create_kg import  search_in_kB, get_article, get_news_links
from newspaper import  ArticleException
from .sentiment import sentiment, sentiment_proba
from .wakeup import collect_data, get_trends, save_kb, save_network_html
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

# from .trends import get_trends



######################### home page ###########################
def home(request):
    print(datetime.now().hour)
    # get trends and save them in a file:
    if datetime.now().hour == 24: 
        trends = get_trends()
        print(trends)
        pickle.dump(trends,open("trends.pkl","wb"))
        collect_data(trends)
    else:
        with open('trends.pkl', 'rb') as handle:
            trends = pickle.load(handle)

    context = {'trends' : trends}
    # if request.method == "POST":
    #     trends = pickle.load(handle)
    #     return kg(request, trends[0])
    return render(request, 'base/home.html', context)


######################### knowledge graph ###########################
def kg(request, trend):
    # t = 0
    name = trend.replace(' ', '_')
    filename = f"/home/sarah/Desktop/web/kb/base/templates/base/{name}"+".html"

   
    kb = pd.read_pickle(f"/home/sarah/Desktop/web/kb/base/knowledge_bases/{name}.p")
    kb_df = pd.DataFrame(kb.sources).T

    entities = pd.DataFrame(kb.relations)["head"].append(pd.DataFrame(kb.relations)["tail"])
    counts = entities.value_counts()
    key_words = []
    for head in entities.unique():
        if counts[head] > 3:
            key_words.append(head)


    print(len(key_words), len(entities))
    context = {
        "trend" : trend,
        "key_words": key_words,
        "graph": f"base/{filename}",
        "sources": zip(kb_df.index, kb_df["article_title"])
    }

    return render(request, f"base/{name}.html", context)
# Create your views here.

def search(request, trend):
    # print(request.form["keyword"])
    filename = "/home/sarah/Desktop/web/kb/base/templates/base/search.html"
    name = trend.replace(' ', '_')

    kb = pd.read_pickle(f"/home/sarah/Desktop/web/kb/base/knowledge_bases/{name}.p")
    selected_entetities = request.POST.getlist("key_word")
    sentences = search_in_kB(kb, selected_entetities, 
                filename)

    # print(sentences)
    # # perform sentiment
    # s = sentiment(sentences)
    # neg_pos = ""
    # if s > 0.5:
    #     neg_pos = "Positive"
    # elif s < 0.5:
    #     neg_pos = "Negative"
    # else:
    #     neg_pos = "Neutral"
    
    
    # with open(filename, 'r') as file:
    #     data = file.readlines()

    #     # details = open("/home/sarah/Desktop/web/kb/base/details.html", "r").read()
    #     #data[-1] = re.sub(r'[\n\t\ ]+', ' ', details)
    #     data[-1] = "{% include 'base/sentiment.html' %}</html>"
    # with open(filename, 'w') as file:
    #         file.writelines( data )

    # context = {"neg_pos":neg_pos,
    # "s":s
    # }
    context ={}
    return render(request, f"base/search.html", context)


###############################################################
###############################################################
###############################################################
def person(request):
    print()
    person = request.POST.getlist('query')[0]
    topic = request.POST.getlist('query')[1]

    query = person + " and " + topic
    name = query.replace(' ', '_')
    
    news_links = get_news_links(query, pages=5, max_links=50)
    articles = []
    for link in news_links:
        if "twitter" in link or "youtube" in link:
            pass    
        else:
            try:
                article = get_article(link).text
                articles.append(article)
            except ArticleException:
                print(f"  Couldn't download article at url {link}")   
    
    print(len(articles), len(article))
    # for link in news_links:
    #     try:
    #         article = get_article(link)
    #         articles.append(article)
    #     except ArticleException:
    #         print(f"  Couldn't download article at url {link}")
    labels, pos_sentences, neg_sentences = sentiment_proba(person,topic,articles)

    sentiment_ = np.average(labels)*100
    if sentiment_> 0.4:
        context = {"query": query,
                    "sentiment": sentiment_,
                    "pos": set(pos_sentences),
                    "neg": set(neg_sentences),
                    }
    else:
        context={"query": query,
                "sentiment": sentiment_,
                "pos": set(pos_sentences),
                "neg": set(neg_sentences),
                  }

    return render(request, f"base/person.html", context)