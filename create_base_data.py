# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:16:08 2020
Starter code to (re)create IMDB TV Series dataset

@author: The Prince
"""

import numpy as np
import pandas as pd

# (1) Load AKA information and aggregate features by tvSeries
akas = pd.read_csv('akas.tsv',sep='\t',low_memory=False)
akas['isOriginalTitle'] = pd.to_numeric(akas['isOriginalTitle'],errors='coerce')
akas = akas.rename(columns={'titleId':'tconst'}).groupby('tconst').agg({'ordering':'count',
                  'title':'nunique','region':'nunique','types':'nunique',
                  'language':'nunique','attributes':'nunique',
                  'isOriginalTitle':'max'})

# (2) Load episodes dataset and aggregate features by tvSeries
episode = pd.read_csv('episode.tsv',sep='\t',low_memory=False)
episode = episode.replace('\\N',np.nan)
episode[["seasonNumber", "episodeNumber"]] = episode[["seasonNumber", "episodeNumber"]].apply(pd.to_numeric)
episode_grouper = episode.groupby('parentTconst').agg({'seasonNumber':'max', 
                                 'episodeNumber':'count'}).reset_index().rename(columns={'parentTconst':'tconst'})

# (3) Load basic attributes datasets and filter down to non-adult tvSeries
basics = pd.read_csv('basics.tsv',sep='\t',low_memory=False)
# Only retain those tv series not adult themed
tv = basics[(basics['titleType'] == 'tvSeries') & 
            (basics['isAdult'] == 0) &
            (basics['genres'] != 'Adult')]
# Replace missing entries with nans, which could be easily removed
tv = tv.replace('\\N',np.nan)
# Replace missing original titles with corresponding primary titles
tv.originalTitle.fillna(tv.primaryTitle,inplace=True)

# (4) Load ratings dataset
ratings = pd.read_csv('ratings.tsv',sep='\t')
ratings['tconst'] = ratings['tconst'].astype('str')

# (5) Merge tv series info with ratings, episodes, and AKA datasets
results1 = pd.merge(tv,ratings,on='tconst',how='left')
results2 = pd.merge(results1,episode_grouper,on='tconst',how='left')
results3 = pd.merge(results2,akas,on='tconst',how='left')

# (6) Drop unnnecessary columns
starter = results3.dropna(subset=['startYear']).drop(columns=['titleType','primaryTitle','isAdult'])

# (7) Title Language Identification
from langid.langid import LanguageIdentifier, model
# Apply langid language identification model to identify languages and 
# probabilities for each tv series title
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
language1 = starter.originalTitle.apply(identifier.classify)
starter.loc[:, 'languagePrediction'] = [x[0] for x in language1]
starter.loc[:, 'languageProbability'] = [x[1] for x in language1]

# (8) Compile and write base datasets to csv file for later loading
starter.to_csv('base_data.csv')


