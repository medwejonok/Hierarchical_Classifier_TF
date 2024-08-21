import nltk
import gzip
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def prepropcess_data(df):
    '''
    '''
    # Y
    if 'keyword' in df.columns:
        text_classes = df['keyword']
        le = LabelEncoder()
        df['keyword'] = le.fit_transform(text_classes)


    # 1) LowerCase
    df['text'] = df['text'].str.lower()

    # 2) Tokenize
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        return [w for w in tokens if w.isalpha()]
    df['text_tokenize'] = df['text'].apply(tokenize)

    # 3) Stop-words
    stop_words = set(stopwords.words('english'))
    df['text_no_stop_words'] = df['text_tokenize'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # 4) Lem
    def lemmatize_word(word):
        lemmatizer = WordNetLemmatizer()
        pos_tag = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos=pos_tag)
        return lemma
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    df['text_lemmatized'] = df['text_no_stop_words'].apply(lambda x: [lemmatize_word(word) for word in x])

    # 5) Final join text
    df['final_text'] = df['text_lemmatized'].apply(lambda x: ' '.join(x))

    

    return df

def get_embeddings_tfidf(df):
    vectorizer = TfidfVectorizer()
    df['text_tfidf'] = df['text_lemmatized'].apply(lambda x: ' '.join(x))
    tf_idf_emb = vectorizer.fit_transform(df['text_tfidf'])
    df['text_tfidf'] = tf_idf_emb.toarray()
    return df


def get_embeddings_gzip(df):
    def gzip_embedding(text):
        compressed_text = gzip.compress(text.encode())
        return len(compressed_text)
    df['text_gzip'] = df['text_lemmatized'].apply(lambda x: gzip_embedding(' '.join(x)))
    return df
