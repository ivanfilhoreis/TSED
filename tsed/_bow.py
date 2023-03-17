from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import re
from keybert import KeyBERT
from typing import List, Union, Tuple
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

class tsed_bow():
    
    def __init__(self, 
                 df_ts: pd.DataFrame, 
                 df_tx: pd.DataFrame, 
                 df_column_tx: str = None,
                 column_concatene: str = 'Date', 
                 drop_duplicates: bool = True,
                 stemmer: bool = False,
                 lowercase: bool = True,
                 empty_text: str = 'no text',
                 ngram_range: Tuple = (1, 1),
                 stop_words = None, 
                 binary: bool = False,
                 max_df: float = 1.0, 
                 min_df: float = 1, 
                 max_features:  int = None, 
                 weighting: str = 'tf-idf', 
                 vocabulary: Union[str, List[str]] = None, 
                 norm: str = 'l2', 
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False, 
                 n_features: int = 2**6, 
                 merge_how: str = 'outer', 
                 ascending: bool = True,
                 kbert_candidates: List[str] = None,
                 kbert_model: str = 'distilbert-base-nli-mean-tokens', 
                 kbert_top_n: int = 5,
                 kbert_min_df: int = 1,
                 kbert_use_maxsum: bool = False,
                 kbert_use_mmr: bool = False,
                 kbert_diversity: float = 0.5,
                 kbert_nr_candidates: int = 20,
                 kbert_vectorizer: CountVectorizer = None,
                 kbert_highlight: bool = False,
                 kbert_seed_keywords: List[str] = None,
                 normalize: bool = True):
        
        self.df_ts = df_ts
        self.df_tx = df_tx
        self.column_concatene = column_concatene
        self.df_column_tx = df_column_tx
        self.drop_duplicates = drop_duplicates
        self.stemmer = stemmer
        self.lowercase = lowercase
        self.empty_text = empty_text
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.binary = binary
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.weighting = weighting
        self.vocabulary = vocabulary
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.n_features = n_features # hashing vectorizer
        self.merge_how = merge_how
        self.ascending = ascending
        self.kbert_candidates = kbert_candidates
        self.kbert_model = kbert_model
        self.kbert_top_n = kbert_top_n
        self.kbert_min_df = kbert_min_df
        self.kbert_use_maxsum = kbert_use_maxsum
        self.kbert_use_mmr = kbert_use_mmr
        self.kbert_diversity = kbert_diversity
        self.kbert_nr_candidates = kbert_nr_candidates
        self.kbert_vectorizer = kbert_vectorizer
        self.kbert_highlight = kbert_highlight
        self.kbert_seed_keywords = kbert_seed_keywords
        self.normalize = normalize
          
    def build_tsed_bow(self):
        
        df = self.concatenate_dfs()
        text = self.Text_PreProcessing(df[self.df_column_tx])
        
        if self.vocabulary == 'keybert':
            self.vocabulary = self.features_keybert(text)
        
        if self.weighting == 'tf':
            tf_vectorizer = CountVectorizer(lowercase = self.lowercase, 
                                            stop_words = self.stop_words, 
                                            ngram_range=self.ngram_range, 
                                            max_df = self.max_df, 
                                            min_df = self.min_df,  
                                            max_features= self.max_features, 
                                            vocabulary=self.vocabulary, 
                                            binary=self.binary)
            tf_fit = tf_vectorizer.fit_transform(text)
            matrix = pd.DataFrame(tf_fit.todense(), columns=tf_vectorizer.get_feature_names_out(), index=df[self.column_concatene])
            
        elif self.weighting == 'tf-idf':
            idf_vectorizer = TfidfVectorizer(lowercase = self.lowercase, 
                                             stop_words = self.stop_words, 
                                             ngram_range = self.ngram_range,
                                             max_df = self.max_df, 
                                             min_df = self.min_df,  
                                             max_features= self.max_features,
                                             vocabulary=self.vocabulary, 
                                             norm=self.norm, 
                                             use_idf=True, 
                                             smooth_idf=self.smooth_idf,
                                             sublinear_tf=self.sublinear_tf)
            idf_fit = idf_vectorizer.fit_transform(text)
            matrix = pd.DataFrame(idf_fit.todense(), columns=idf_vectorizer.get_feature_names_out(), index=df[self.column_concatene])
            
        elif self.weighting == 'hashing':
            has_vectorizer = HashingVectorizer(lowercase = self.lowercase, 
                                               stop_words = self.stop_words, 
                                               ngram_range=self.ngram_range,
                                               n_features = self.n_features, 
                                               norm=self.norm, 
                                               binary=self.binary)
            has_fit = has_vectorizer.fit_transform(text)
            matrix = pd.DataFrame(has_fit.todense(), columns=list(range(0, self.n_features)), index=df[self.column_concatene])
            
        else:
            print('Error: Chosse a correct option: tf, tf-idf or hashing')
        
        
        df.drop(columns=[self.df_column_tx], inplace=True)
        df_cnt = df.set_index('Date').join(matrix)
        
        return df_cnt
    
    def Text_PreProcessing(self, df):
        texts = []
        for row in range(0, len(df)):
            txt = str(df.iloc[row])      
            txt = re.sub('[0-9][^w]', '' , txt)
            texts.append(txt)
          
        if self.stemmer:
            _, texts = self.Stemming(texts, 2)
            
        return texts
    
    def Stemming(self, vocab, op):
        stemmer = SnowballStemmer('english')
        
        if op == 1:
            texts = None
            vocabulary = set(stemmer.stem(d) for d in vocab)
        elif op == 2:
            texts = []
            vocabulary = None
            for txt in vocab:
                s = str(txt).lower()
                tokens = nltk.word_tokenize(s)
                estemas = [stemmer.stem(token) for token in tokens]
                                
                s = ""
                for token in estemas:
                    s += token+" "
          
                texts.append(s.strip())
      
        return vocabulary, texts
      
    def features_keybert(self, text):
                 
        kw_model = KeyBERT(self.kbert_model)
        keys = list()
        result = 0
        
        for txt, prog in zip(text, tqdm(range(0, len(text)))):
            result += prog
            if txt == 'no texts':
                continue
            
            keywords = kw_model.extract_keywords(txt,
                                                 candidates = self.kbert_candidates,
                                                 keyphrase_ngram_range = self.ngram_range,
                                                 stop_words = self.stop_words,
                                                 top_n = self.kbert_top_n,
                                                 min_df = self.kbert_min_df,
                                                 use_maxsum = self.kbert_use_maxsum,
                                                 use_mmr = self.kbert_use_mmr,
                                                 diversity = self.kbert_diversity,
                                                 nr_candidates = self.kbert_nr_candidates,
                                                 vectorizer = self.kbert_vectorizer,
                                                 highlight = self.kbert_highlight,
                                                 seed_keywords = self.kbert_seed_keywords)

            keys.append(keywords)
            
        vocab = []
        for k in keys:
            for (kw, vl) in k:
                vocab.append(kw)

        if(self.stemmer):
            vocabulary, _ = self.Stemming(vocab, 1)
        else:
            vocabulary = set(d for d in vocab)
        
        print(vocabulary)
        return vocabulary
    
    def concatenate_dfs(self):
        
        self.df_ts[self.column_concatene] = pd.to_datetime(self.df_ts[self.column_concatene])
        self.df_ts[self.column_concatene].dropna(inplace=True)
        self.df_ts.sort_values(self.column_concatene, ascending=self.ascending, inplace=True)
        
        self.df_tx = self.df_tx.loc[:, [self.column_concatene, self.df_column_tx]]
        self.df_tx[self.column_concatene] = pd.to_datetime(self.df_tx[self.column_concatene])
        self.df_tx[self.column_concatene].dropna(inplace=True)
        self.df_tx.sort_values(self.column_concatene, ascending=self.ascending, inplace=True)
    
        df_cnt = pd.merge(self.df_ts, self.df_tx, on=self.column_concatene, how=self.merge_how)
        df_cnt[self.df_column_tx].replace({np.nan: self.empty_text}, inplace=True)
        if (self.drop_duplicates):
            df_cnt.drop_duplicates(subset=self.column_concatene, keep="first", inplace=True)
        
        return df_cnt
