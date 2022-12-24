<h1> Natural Language Processing & Twitter: 2020 Elections Topic Modeling</h1>

<p align="center">

<h2>Description</h2>

The goal is to test out different Topic Modeling algorithms, namely LDA (Latent Dirichlet Allocation) and NMF (NonNegative Matrix Factorization), which can be found within the Gensim package.

💡 Inspired from [This research paper](https://link.springer.com/article/10.1007/s42001-021-00117-9)<br/>
💡 This is just a test for the incoming 2024 elections.<br/>

<br />

<h2>Python Libraries and Utilities</h2>

 - <b>pandas</b>
 - <b>nltk</b>
 - <b>re</b>
 - <b>gensim</b>
 - <b>tqdm</b>
 - <b>pyLDAvis</b>
 
<br />

<h2>Environments Used </h2>

- <b>macOS Monterey</b>

<br />

<h2>Project walk-through:</h2>

<br />
 
<h3> Setup </h3>

**Step 1. Load Libraries** <br/>

```py
import pandas as pd

import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import tqdm

import gensim
```

<br />

**Step 2. Load Data** <br/>

*NOTE: the data was sourced from Kaggle, and it can be found in the repo*

```py
dft = pd.read_csv(FILE PATH TO 'hashtag_donladtrump.csv',lineterminator='\n')
dfb = pd.read_csv(FILE PATH TO 'hashtag_joebiden.csv',lineterminator='\n')

df = pd.concat([dft,dfb],ignore_index=True)
```

<br />

<h3> Text Pre-Processing </h3>

**Step 3. Define Cleaning Function** <br/>

```py
def tweet_preprocess(tweet):
    # lowercase
    tweet = tweet.lower()
    
    #keep only alphabets
    tweet = re.sub(r'[^a-zA-Z]+', ' ', tweet)
    tweet = tweet.replace('\n', '')
    
    #tokenization
    word_list = nltk.word_tokenize(tweet)    
    stopwords_list = nltk.corpus.stopwords.words('english')
    stopwords_list.extend(['trump','realdonaldtrump','thank','trump','presid','america','american','fjv'])
    word_list = [word for word in word_list if word not in stopwords_list]
    
    #small words removal
    word_list = [word for word in word_list if len(word)>3]
    
    #stemmer and lemmatizer
    porter_stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    word_list = [porter_stemmer.stem(word) for word in word_list]
    word_list = [lemmatizer.lemmatize(word) for word in word_list]
    
    tweet = ' '.join(word_list)
    
    return tweet
```

```p
tqdm.tqdm.pandas()
df['tweet_tokenized'] = df['tweet'].progress_apply(lambda x:tweet_preprocess(str(x)))

performance_metrics = pd.DataFrame(columns=['feature-extraction','clustering-algo','c_v','c_umass','topics'])
```

<br />

**Step 4. Algo 1: TF-IDF** <br/>

```p
documents = df['tweet_tokenized'].str.split()

dictionary = gensim.corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=20000)

tfidf = gensim.models.TfidfModel(dictionary=dictionary)

corpus = [dictionary.doc2bow(document) for document in documents]

corpus_tfidf = list(tfidf[corpus])
```

<br />

**Step 5. Topic Modeling** <br/>

```p
#Parameters
EPOCHS = 205
TOPICS = 10
CHUNK_SIZE = 1000
WORKERS = 7
EVAL_PERIOD = 10
ALPHA = 0.01
BETA = 0.9

lda = gensim.models.ldamodel.LdaModel(
    corpus = corpus_tfidf,
    num_topics = 10,
    id2word = dictionary,
    chunksize=CHUNK_SIZE, passes=EPOCHS, 
    eval_every = EVAL_PERIOD, 
    per_word_topics=True
    )
    
topics_lda = lda.print_topics()

coherence_cv = gensim.models.CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='c_v').get_coherence()
coherence_cumass = gensim.models.CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='u_mass').get_coherence()

performance_metrics = performance_metrics.append({'feature-extraction':'tf-idf', 
                                                  'clustering-algo':'LDA',
                                                  'c_v':coherence_cv,
                                                  'c_umass':coherence_cumass,
                                                  'topics':topics_lda}, 
                                                   ignore_index=True)
```

<br />

**Step 6. Algo 2: NMF** <br/>

```p
nmf = gensim.models.Nmf(corpus=corpus_tfidf, 
                        num_topics=TOPICS, 
                        id2word=dictionary, 
                        chunksize=CHUNK_SIZE, passes=EPOCHS, 
                        eval_every=EVAL_PERIOD, 
                        minimum_probability=0, 
                        kappa=1
                       )
                       
topics_nmf = nmf.print_topics()

coherence_cv = gensim.models.CoherenceModel(model=nmf, texts=documents, dictionary=dictionary, coherence='c_v').get_coherence()
coherence_cumass = gensim.models.CoherenceModel(model=nmf, texts=documents, dictionary=dictionary, coherence='u_mass').get_coherence()

performance_metrics = performance_metrics.append({'feature-extraction':'tf-idf', 
                                                  'clustering-algo':'NMF',
                                                  'c_v':coherence_cv,
                                                  'c_umass':coherence_cumass,
                                                  'topics':topics_nmf}, 
                                                   ignore_index=True)
```

<br />

**Step 7. Evaluate Algorithms** <br/>

```p
performance_metrics

mean_perf = performance_metrics.groupby('clustering-algo')[['c_v','c_umass']].mean()

performance_metrics.to_csv('/Users/matteo-stelluti/Desktop/Assistant Research/2020 Elections/test_performance.csv')
print(mean_perf)
```

<img src="https://i.imgur.com/n7IXY4m.png" height="40%" width="40%" alt="Algo comparison"/> <br />

**Step 7. Topics Visualization** <br/>

```p
import pyLDAvis
import pyLDAvis.gensim_models as gensimviz
pyLDAvis.enable_notebook() #This is only needed on Jupyter Notebook

gensimviz.prepare(lda,corpus,dictionary)
```

<img src="https://i.imgur.com/koNOHdQ.png" height="80%" width="80%" alt="Topics"/> <br />


</p>



<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
