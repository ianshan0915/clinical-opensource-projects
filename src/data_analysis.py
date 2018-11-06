# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed May 30 2018

@author: ianshan0915

Use GitHub REST API v3 to query open sourced repos related to clinical/medical
"""

import requests
import itertools
import json

import pandas as pd
import numpy as np

from collections import Counter

from langdetect import detect

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions

import textrazor
import nltk
import spacy

import plotly
plotly.tools.set_credentials_file(username='ianshan0915', api_key='VajGKj0SOaXAGGcqJD8y')
plotly.tools.set_config_file(world_readable=True)

import plotly.plotly as py
import plotly.graph_objs as go

from scipy import stats

from gensim import corpora
import gensim
import pyLDAvis.gensim

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def load_data(filter_lang):
  """
  load the repos data collected from Github API
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos.csv')

  # preprocess the repos' language column
  repos.loc[repos['language'].isin(['CSS','HTML']), 'language'] = 'CSS/HTML'
  repos.loc[repos['language'].isin(['TypeScript']), 'language'] = 'JavaScript'
  if filter_lang:
    repos = repos.loc[repos.language.notnull()] # repos with no language(s) indicated are removed

  return repos

def check_lang_trends():
  """
  to uncover the evolution of progoramming languages  
  """

  df = load_data(True)
  df = filter_irrelevant(df)
  groupedTops = df.groupby(['language']).id.agg('count').to_frame('count').reset_index().sort_values(by=['count'], ascending=False)
  top10s = groupedTops.head(10).language.values
  df.loc[~df['language'].isin(top10s), 'language'] = 'Others'
  df['created_at'] = pd.to_datetime(df['created_at'])
  df['created_year'] = df['created_at'].map(lambda x: x.strftime('%Y'))
  df['owner'] = df.full_name.apply(lambda x: x.split('/')[0])
  grouped = df.groupby(['created_year','language']).id.agg('count').to_frame('count').reset_index()
  groupedOwners = df.groupby(['owner','owner_type']).id.agg('count').to_frame('count').reset_index().sort_values(by=['count'], ascending=False)
  piv_df = grouped.pivot(index='language', columns='created_year', values='count')
  piv_df = piv_df.reset_index().fillna(0)
  # groupedTops.head(10).to_csv('~/Documents/top10_languages.csv', index=False)
  groupedOwners.head(10).to_csv('~/Documents/owners_types.csv', index=False)
  piv_df.to_csv('~/Documents/languages_over_time.csv', index=False)

  return piv_df

def get_top_owners(df):
  """
  obtain top contributors to the clinical open source projects
  """

  pass

  return None

def filter_irrelevant(df):
  """
  filter out irrelevant repositories, like repos with 'patiently' in the description
  """

  df_irrelevant = df.loc[(df.description.str.contains('patiently')) | (df.description.str.contains('be patient')) | (df.description.str.contains('doctor who')) ]
  df_clean = df.loc[~df.index.isin(df_irrelevant.index)]

  return df_clean

def ibmwaston_nlu(repo_description):
  """
  use IBM waston-NLU to process texutal input
  """

  usernm = '1a5fba40-ab91-4b36-92a0-f957b6abeb0f'
  passwd = 'jut4J5KJaoXe'
  natural_language_understanding = NaturalLanguageUnderstandingV1(username= usernm, password= passwd, version='2018-03-16')
  text = repo_description.encode('utf-8')
  print(text)
  response = natural_language_understanding.analyze(
    text= repo_description,
    language = 'en',
    features=Features(
      keywords=KeywordsOptions(
        limit=5)))

  keywords = [item['text'] for item in response['keywords'] if item['relevance']>0.5]
  kw_str = ', '.join(keywords)

  return kw_str

def text_razor(repo_description):
  """
  Use TextRazor to process textual input
  """

  textrazor.api_key = '5f6331ac5ecb61dfe6e57d9706eeb4f9e7bceaa82a4a37b128cb0201'
  textrazor.language_override = 'en'
  client = textrazor.TextRazor(extractors=['entities'])
  response = client.analyze(repo_description)
  phrases = response.entities()
  keywords = [item.matched_text for item in phrases]
  kw_str = ', '.join(keywords)

  return kw_str

def dandelion(repo_description):
  """
  use Dandelion API to process texual input
  """
  apiurl = 'https://api.dandelion.eu/datatxt/nex/v1'
  apikey = '61fe441ac2f04418beb47add443d57b8'
  headers = {'content-type': 'application/x-www-form-urlencoded'}
  apikey_str = "token=" + apikey + "&"
  repo_description = "text="+ repo_description + "&"
  lang_str = "lang=" + "en" + "&"
  incl_str = "include=" + "types,categories"

  payload = apikey_str + repo_description + lang_str + incl_str
  response = requests.request("POST", apiurl, data=payload, headers=headers)
  resp_json = response.json()
  keywords = [item['spot'] for item in resp_json['annotations'] if item['confidence']>0.5]
  kw_str = ', '.join(keywords)

  return kw_str

def fetch_topics(url):
  """
  fetch topics given a repo url
  """
  print(url)
  headers = {'Accept': 'application/vnd.github.mercy-preview+json', 'Authorization': 'token 0871d9a1ca33296d189644fd2cafa9fe45d4e62b'}
  response = requests.request("GET", url, headers=headers)
  resp_json = response.json()
  if 'topics' in resp_json.keys():
    topics = resp_json['topics']
  else:
    topics = []
  if len(topics) >0:
    topics_str = ', '.join(topics)
  else:
    topics_str = ''

  return topics_str

def gen_topics():
  """
  get topics for github repositories given their api url
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_keywords.csv')
  repos = repos.loc[repos['size']>0]
  dfs = np.array_split(repos, 4)
  # repos = repos.head(200)
  # dfs[0]['topics'] = dfs[0].url.apply(lambda x: fetch_topics(x))
  # dfs[0].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics_1.csv', index=False)
  # dfs[1]['topics'] = dfs[1].url.apply(lambda x: fetch_topics(x))
  # dfs[1].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics_2.csv', index=False)
  # dfs[2]['topics'] = dfs[2].url.apply(lambda x: fetch_topics(x))
  # dfs[2].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics_3.csv', index=False)
  dfs[3]['topics'] = dfs[3].url.apply(lambda x: fetch_topics(x))
  dfs[3].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics_4.csv', index=False)
  
  return None

def check_lang(repo_description):
  """
  check if any invalid unicode
  """

  try:
    lang = detect(repo_description)
  except UnicodeError:
    print(repo_description)

  return lang

def gen_keywords():
  """
  extract keywords from the repo's description using external NLP apis: ibm watson, textrazor, dandelion
  """

  df = load_data(False)
  df = filter_irrelevant(df)
  dfs = np.array_split(df, 3)
  dfs[0]['watson_keywords'] = dfs[0].description.apply(lambda x: ibmwaston_nlu(x) if len(x.split())>3 else x)
  dfs[0].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_1.csv', index=False)
  dfs[1]['watson_keywords'] = dfs[1].description.apply(lambda x: ibmwaston_nlu(x) if len(x.split())>3 else x)
  dfs[1].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_2.csv', index=False)
  dfs[2]['watson_keywords'] = dfs[2].description.apply(lambda x: ibmwaston_nlu(x) if len(x.split())>3 else x)
  dfs[2].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_3.csv', index=False)
  # df['textrazor_keywords'] = df.description.apply(lambda x: text_razor(x))
  # df['dandelion_keywords'] = df.description.apply(lambda x: dandelion(x))

  return None

def keywords_summary(keywords):
  """
  get a list of keywords that appear frequently
  """

  # repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_keywords.csv')
  # repos = repos.loc[repos['size']>0]
  # keywords = [str(item).split(',') for item in repos.watson_keywords.values]
  # p = nltk.SnowballStemmer('english')
  keywords = [val for sublist in keywords for val in sublist]
  counters = Counter(keywords)
  keywords_counter = pd.DataFrame(np.array([list(counters.keys()), list(counters.values())]).T, columns=['keyword','count'])
  keywords_counter[['count']] = keywords_counter[['count']].astype(int)
  keywords_counter = keywords_counter.sort_values(by=['count'], ascending=False)
  keywords_counter.to_csv('~/Documents/keywords_count.csv', index=False)
  
  return None

def process_keywords(spacy_nlp, words, type):
  """
  process keywords with spacy 
    -- type: tag type, NN or adj
  """

  result = spacy_nlp(words)
  if type =='chunk':
    text = ' '.join([chunk.text for chunk in result])
  else:
    text = ' '.join([token.lemma_ for token in result if type in token.tag_])
  
  return text

def cal_similarity(spacy_nlp):
  """
  spacy_nlp: nlp pipeline with a specified db 
  keywords: a list of keywords
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
  repos['watson_keywords'] = repos.watson_keywords.str.replace(r'\bapp\b', 'application', case=False)
  repos['watson_keywords'] = repos.watson_keywords.str.replace(r'\bpatient\b', 'patients', case=False)
  keywords = [str(item).split(',') for item in repos.watson_keywords.values]
  keywords = [val.strip().lower() for sublist in keywords for val in sublist]
  counters = Counter(keywords)
  keywords_counter = pd.DataFrame(np.array([list(counters.keys()), list(counters.values())]).T, columns=['keyword','count'])
  keywords_counter[['count']] = keywords_counter[['count']].astype(int)
  keywords_counter = keywords_counter.sort_values(by=['count'], ascending=False)  
  keywords_top = keywords_counter.head(500)
  keywords_rest = keywords_counter.loc[~keywords_counter.index.isin(keywords_top.index)]
  keywords_lst = keywords_top.keyword.values
  keywords_rest = keywords_rest.keyword.values
  keywords_nlp = [spacy_nlp(keyword) for keyword in keywords_lst]
  keywords_rest_nlp = [spacy_nlp(keyword) for keyword in keywords_rest]
  similarity_array = []
  similarity_array_rest = []
  print('preparing the df....')
  for keyword1, keyword2 in itertools.combinations(keywords_nlp, 2):
    similarity_score1 = keyword1.similarity(keyword2)
    nn_token1_lst = [token for token in keyword1 if token.tag_ in ['NN','NNS','VBG']]
    nn_token2_lst = [token for token in keyword2 if token.tag_ in ['NN','NNS','VBG']]
    if len(nn_token1_lst) >0 and len(nn_token2_lst)>0:
      nn_token1 = nn_token1_lst[len(nn_token1_lst)-1]
      nn_token2 = nn_token2_lst[len(nn_token2_lst)-1]
      similarity_score2 = nn_token1.similarity(nn_token2)
    else:
      similarity_score2=0
    similarity_row = [keyword1, keyword2, similarity_score1, similarity_score2]
    similarity_array.append(similarity_row)
  print('preparing the rest df....')
  for keywd1, keywd2 in itertools.product(keywords_rest_nlp, keywords_nlp):
    similarity_score1 = keywd1.similarity(keywd2)
    if similarity_score1>=0.75:
      nn_token1_lst = [token for token in keywd1 if token.tag_ in ['NN','NNS','VBG']]
      nn_token2_lst = [token for token in keywd2 if token.tag_ in ['NN','NNS','VBG']]
      if len(nn_token1_lst) >0 and len(nn_token2_lst)>0:
        nn_token1 = nn_token1_lst[len(nn_token1_lst)-1]
        nn_token2 = nn_token2_lst[len(nn_token2_lst)-1]
        similarity_score2 = nn_token1.similarity(nn_token2)
      else:
        similarity_score2=0
      similarity_row = [keywd2, keywd1, similarity_score1, similarity_score2]
      similarity_array_rest.append(similarity_row)
  df = pd.DataFrame(similarity_array, columns=['keyword1','keyword2','score1','score2'])
  df_rest = pd.DataFrame(similarity_array_rest, columns=['keyword1','keyword2','score1', 'score2'])
  df_rest['score'] = (df_rest['score1']*0.75 + df_rest['score2']*0.25).round(2)
  df['score'] = (df['score1']*0.75 + df['score2']*0.25).round(2)  
  df_rest.to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/rest_similarity_score.csv', index=False)
  df.to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/similarity_score.csv', index=False)

  return df_rest, df

def fetch_map(key1, key2, score1, score2, type):
  """
  """

  if len(key1.split())==1 and len(key2.split())==1 and type=='eq_val':
    return {'keyword': key1, 'score': score1}
  elif len(key1.split())==1 and len(key2.split())>1 and type=='tag':
    return {'keyword': key1, 'score': score1}
  elif len(key1.split())>1 and len(key2.split())>1 and score1>=0.85 and score2> 0.6 and type=='eq_val':
    return {'keyword': key1, 'score': score1}
  elif len(key1.split())>1 and len(key2.split())>1 and score1>=0.8 and type=='tag':
    return {'keyword': key1, 'score': score1}

def keywords_mapping(type):
  """
  keywords: dataframe with keywords similarity scores
  """

  if type == 'top':
    keywords = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/similarity_score.csv')
  else:
    keywords = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/rest_similarity_score.csv')
  keywords_2 = keywords.keyword2.unique()
  keyword_maps = []
  for keyword in keywords_2:
    df_tmp = keywords.loc[(keywords['keyword2']==keyword) & ((keywords['score']>=0.70) | (keywords['score1']>=0.80)), ('keyword1','keyword2', 'score1', 'score2')]
    map_keyword = {'val': keyword, 'tag': [], 'eq_val': []}
    arr_tmp = df_tmp.values
    if len(arr_tmp) >0:
      vals_tag = [fetch_map(key1, key2, score1, score2, 'tag') for key1, key2, score1, score2 in arr_tmp]
      vals_eq = [fetch_map(key1, key2, score1, score2, 'eq_val') for key1, key2, score1, score2 in arr_tmp]
      map_keyword['tag'] = [val for val in vals_tag if val is not None]
      map_keyword['eq_val'] = [val for val in vals_eq if val is not None]
      if len(map_keyword['eq_val'])>1:
        map_keyword['eq_val'] = sorted(map_keyword['eq_val'], key= lambda k:k['score'], reverse=True)[:1]
      if len(map_keyword['tag'])>3:
        map_keyword['tag'] = sorted(map_keyword['tag'], key= lambda k: k['score'], reverse=True)[:3]
      map_keyword['eq_val'] = [item['keyword'] for item in map_keyword['eq_val']]
      map_keyword['tag'] = [ item['keyword'] for item in map_keyword['tag']]
    keyword_maps.append(map_keyword)
    keywords = keywords.loc[keywords['keyword2']!=keyword]
  keyword_eq = [item for item in keyword_maps if len(item['eq_val'])>0]
  # with open('/Users/ianshen/Documents/rest_keyword_maps.json', 'w') as fp1:
  #   print(len(keyword_maps))
  #   json.dump(keyword_maps, fp1, indent=4)
  # with open('/Users/ianshen/Documents/rest_keyword_maps_eq.json', 'w') as fp2:
  #   print(len(keyword_eq))
  #   json.dump(keyword_eq, fp2, indent=4)

  return keyword_maps, keyword_eq

def replace_eq(item, keyword_eqs):
  """
  """
  
  filtered = list(filter(lambda x: x['val']== item, keyword_eqs))
  if len(filtered)>0:
    item = filtered[0]['eq_val'][0]

  return item

def process_keyword_map(keyword_map, keyword_eqs):
  """
  """

  keyword_map['tag'] = [replace_eq(item, keyword_eqs) for item in keyword_map['tag']]
  keyword_map['eq_val'] = [replace_eq(item, keyword_eqs) for item in keyword_map['eq_val']]

  return keyword_map

def keywords_normalization():
  """
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
  keywords = [str(item).split(',') for item in repos.watson_keywords.values]
  keywords = [val.strip().lower() for sublist in keywords for val in sublist]
  nlp = spacy.load('en_core_web_sm')
  keywords_nn = [process_keywords(nlp, val, 'NN') for val in keywords]
  keywords_adj = [process_keywords(nlp, val, 'JJ') for val in keywords]
  df = pd.DataFrame({'keywords': keywords, 'nouns': keywords_nn, 'adjs': keywords_adj})
  grouped = df.groupby(['keywords']).count().reset_index()
  df.to_csv('~/Documents/keywords_processed.csv', index=False)

  return None

def keywords_normalize(keywords, keyword_maps):
  """
  normailize keywords by matching them based on semantic similarity
  """

  norm_keywords = []
  for keyword in keywords:
    keyword = keyword.strip().lower()
    keyword_map = list(filter(lambda x: x['val']== keyword, keyword_maps))
    if len(keyword_map)>0:
      keyword_tags = keyword_map[0]['tag']
      if len(keyword_tags) > 1:
        keyword_tags = keyword_tags[:1]
      keyword_eq = keyword_map[0]['eq_val']
      if len(keyword_eq) >0:
        tmp_keywords = keyword_eq + keyword_tags
      else:
        tmp_keywords = [keyword] + keyword_tags
    else:
      tmp_keywords = [keyword]
    norm_keywords = norm_keywords + tmp_keywords
  norm_keywords = np.unique(norm_keywords).tolist()

  return norm_keywords

def topic_clustering(keyword_maps):
  """
  perform topic modelling on description data 
  """
  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
  # repos = repos.loc[(repos['created_at']>'2015-01-01T00:00:00Z') & (repos['created_at']<='2016-01-01T00:00:00Z')]
  # repos = repos.loc[(repos['created_at']<='2015-01-01T00:00:00Z')]
  repos['created_at'] = pd.to_datetime(repos['created_at'])
  repos['created_year'] = repos.created_at.dt.year
  repos['watson_keywords'] = repos.watson_keywords.str.replace(r'\bapp\b', 'application', case=False)
  repos['watson_keywords'] = repos.watson_keywords.str.replace(r'\bpatient\b', 'patients', case=False)
  keywords_raw = [str(item).split(',') for item in repos.watson_keywords.values]
  # p = nltk.SnowballStemmer('english')
  # with open('/Users/ianshen/Documents/github/clinical-opensource-projects/data/keyword_maps.json', 'rb') as fp:
  #   keyword_maps = json.load(fp)
  keywords_norm = [keywords_normalize(sublist, keyword_maps) for sublist in keywords_raw]
  # keywords_summary(keywords_norm)
  # pd.DataFrame({'raw': keywords_raw, 'norm':keywords_norm}).to_csv('~/Documents/raw_norm_keywords.csv', index=False)
  # topic modelling with lda
  dictionary = corpora.Dictionary(keywords_norm)
  corpus = [dictionary.doc2bow(text) for text in keywords_norm]
  print(len(corpus))
  NUM_TOPICS = 10
  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, alpha=0.001, id2word=dictionary, passes=10)

  times = repos.created_year.values
  keywords_probs = [ldamodel[keyword] for keyword in corpus]
  topics_probs = [ assemble_dict(prob, year, text) for prob, year, text in zip(keywords_probs, times, keywords_norm)]
  print('calulate the topic trends over time...')
  cal_topics_trends(topics_probs)
  with open('/Users/ianshen/Documents/topic_probs.json', 'w') as fp1:
    json.dump(topics_probs, fp1, indent=4) 

  ldamodel.save('/Users/ianshen/Documents/github/clinical-opensource-projects/results/lda_1.model')
  lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
  pyLDAvis.show(lda_display)

  return None

def assemble_dict(prob, year, text):
  prob_tmp = []
  if len(prob)>0:
    for item in prob:
      prob_tmp.append({str(item[0]):float(item[1])})
  return {'year': int(year), 'probs': prob_tmp, 'keywords':text}

def cal_spearman_cof():
  """
  calculate the Spearman's correlation coefficient for hypothesis testing
  """

  repos = pd.read_csv('')
  repos = repos.loc[repos['language'].notnull()]
  stats.spearmanr(repos.readme_size.values, repos.stargazers_count.values)

  return None

def radar_chart_owners():
  """
  generate radar chart for different types of owners, showing the difference btw organizational and individual contributors
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
  repos_metrics = repos[['forks', 'stargazers_count', 'open_issues', 'size', 'readme_size','has_pages', 'owner_type']]
  grouped1 = repos_metrics[['forks', 'stargazers_count', 'open_issues', 'size', 'readme_size', 'owner_type']].groupby('owner_type').agg('mean').reset_index()
  grouped2 = repos_metrics[['has_pages', 'owner_type']].groupby('owner_type').agg(['sum', 'count']).reset_index()
  grouped2.columns = grouped2.columns.droplevel()
  # grouped3 = repos_metrics[['has_wiki','owner_type']].groupby('owner_type').agg(['sum', 'count']).reset_index()
  # grouped3.columns = grouped3.columns.droplevel()
  grouped1['has_pages_percent'] = grouped2['sum']/grouped2['count'] * 100
  # grouped1['has_wiki_percent'] = grouped3['sum']/grouped3['count'] * 10
  grouped1['size'] = grouped1['size']/1000
  grouped1['readme_size'] = grouped1['readme_size']/1000
  grouped1 = grouped1.round(2)
  df = grouped1.drop(['owner_type'], axis=1)
  metrics = list(df)
  num_metrics = len(metrics)

  data = [
      go.Scatterpolar(
        r = df.values[0],
        theta = metrics,
        fill = 'toself',
        name = 'Organization'
      ),
      go.Scatterpolar(
        r = df.values[1],
        theta = metrics,
        fill = 'toself',
        name = 'Individual developer'
      )
  ]

  layout = go.Layout(
    polar = dict(
      radialaxis = dict(
        visible = True,
        range = [0, 30]
      )
    ),
    showlegend = False
  )

  fig = go.Figure(data=data, layout=layout)
  py.plot(fig, filename = "radar_chart_repos", auto_open=True)

  return None

def cal_topics_trends(topics_probs):

  tm_ranges = [2000, 2014, 2015, 2016, 2017, 2018, 2019]
  percs = []
  for ind in range(6):
    # print(tm_ranges[ind], tm_ranges[ind+1])
    tm_topics = [x['probs'] for x in topics_probs if x['year']>=tm_ranges[ind] and x['year']<tm_ranges[ind+1]]
    tm_probs = [val for sublist in tm_topics for val in sublist]
    t_1 = [ k.get('0') for k in tm_probs if k.get('0')]
    t_2 = [ k.get('1') for k in tm_probs if k.get('1')]
    t_3 = [ k.get('2') for k in tm_probs if k.get('2')]
    t_4 = [ k.get('3') for k in tm_probs if k.get('3')]
    t_5 = [ k.get('4') for k in tm_probs if k.get('4')]
    t_6 = [ k.get('5') for k in tm_probs if k.get('5')]
    t_7 = [ k.get('6') for k in tm_probs if k.get('6')]
    t_8 = [ k.get('7') for k in tm_probs if k.get('7')]
    t_9 = [ k.get('8') for k in tm_probs if k.get('8')]
    t_10 = [ k.get('9') for k in tm_probs if k.get('9')]

    topic_1 = sum((1 for x in t_1))/len(tm_topics)*100
    topic_2 = sum((1 for x in t_2))/len(tm_topics)*100
    topic_3 = sum((1 for x in t_3))/len(tm_topics)*100
    topic_4 = sum((1 for x in t_4))/len(tm_topics)*100
    topic_5 = sum((1 for x in t_5))/len(tm_topics)*100
    topic_6 = sum((1 for x in t_6))/len(tm_topics)*100
    topic_7 = sum((1 for x in t_7))/len(tm_topics)*100
    topic_8 = sum((1 for x in t_8))/len(tm_topics)*100
    topic_9 = sum((1 for x in t_9))/len(tm_topics)*100
    topic_10 = sum((1 for x in t_10))/len(tm_topics)*100
    percs.append([topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, topic_8, topic_9, topic_10])
  col_names = ['topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10']
  percs_df = pd.DataFrame(percs, columns=col_names)
  percs_df.to_csv('~/Documents/percs_df.csv', index=False)

  return None

def main():
  """do a list of tasks"""

  # langs_trends = check_lang_trends()
  # repos = gen_keywords()
  # repos = gen_topics()
  # radar_chart_owners()
  # keywords_normalization()
  # print('calculating similarity ....')
  # nlp = spacy.load('en_core_web_lg')
  # rest_keywords, keywords = cal_similarity(nlp)
  # print('done with calculation, move on to the next step!')
  # keyword_maps, keyword_eqs = keywords_mapping('top')
  # rest_keyword_maps, rest_keyword_eqs = keywords_mapping('rest')
  # keyword_maps = keyword_maps + rest_keyword_maps
  # with open('/Users/ianshen/Documents/keyword_maps.json', 'r') as fp1:
  #   # json.dump(keyword_maps, fp1, indent=4)  
  #   keyword_maps = json.load(fp1)
  # with open('/Users/ianshen/Documents/keyword_maps_eq.json', 'r') as fp3:
  #   # json.dump(keyword_maps, fp1, indent=4)  
  #   keyword_eqs = json.load(fp3)    
  # keyword_maps = [process_keyword_map(item, keyword_eqs) for item in keyword_maps]

  print('start to cluster topics...')
  with open('/Users/ianshen/Documents/keyword_maps_test.json', 'r') as fp2:
    keyword_maps = json.load(fp2)
  topic_clustering(keyword_maps)

  # print(repos.head())
  # repos.to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_keywords.csv', index=False)

if __name__ == "__main__":
  main()