# TSED
Time Series representation models Enriched with text Domain resources (TSED)

Forecasting models generally use quantitative time-series data. However, external factors can influence data in time-series, such as political events, economic crises, government macroeconomic policy, and the foreign exchange market. This information is not explicit in the time-series and can influence the prediction of the variable values. Textual data can be a source of knowledge about external factors and potentially useful for time-series forecasting models.

# 1. Installation

Installation and use can be done as:

```
!pip install KeyBERT
!pip install git+https://github.com/ivanfilhoreis/TSED.git
```
# 2. Usage

```
from tsed import tsed_bow
import pandas as pd

df1 = pd.read_csv('ts.csv', header=0, delimiter=",") # Time-series file
df2 = pd.read_excel('news.xlsx') # Textual data

df3 = tsed_bow(df1, 
               df2,
               df_column_tx = 'News', # column of the df2 that contains the text. 
               column_concatene = 'Date', #index/column used to merge df1 and df2. Default: 'Date'
               weighting='tf-idf', 
               ngram_range = (1,1), 
               stemmer=True, 
               stop_words='english', 
               min_df=0.20, 
               max_df=0.80)

df3.build_tsed_bow()

```

Use [KeyBert](https://github.com/ivanfilhoreis/KeyBERT/) to get important words from texts and assign BoW vocabulary/terms.

```
df4 = tsed_bow(df1, 
               df2,
               df_column_tx =  'News', 
               column_concatene = 'Date', 
               weighting='tf-idf', 
               ngram_range = (1,1), 
               stemmer=True, 
               stop_words='english', 
               vocabulary='keybert')

df4.build_tsed_bow()

```

[Click here](https://colab.research.google.com/drive/1hQxVviNUwvWvKnytS9fg3ecXmkawMxqs?usp=sharing) to view others time-series representations models and results.

# 3. Repositórios

Below, you can find some resources that were used for the creation of TSED. 

*[KeyBert - https://github.com/ivanfilhoreis/KeyBERT/](https://github.com/ivanfilhoreis/KeyBERT/)
*[Vectorizers - sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
