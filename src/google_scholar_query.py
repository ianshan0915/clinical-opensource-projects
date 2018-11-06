# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed Aug 27 2018

@author: ianshan0915 (https://github.com/ianshan0915)

Use scholar.py from Christian Kreibich (https://github.com/ckreibich/scholar.py) 
to query papers for each of the extracted clinical/medical related Github repository
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from scholar import ScholarConf, SearchScholarQuery, ScholarQuerier, ScholarSettings

import time
import random
import json
import math
import re

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

def load_repos():
    """
    load the repository and do some basic filtering
    """

    repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
    # repos = repos.loc[repos['forks']>0]
    repos = repos.loc[(repos['language'].notnull())  & ((repos['forks']>0)|(repos['stargazers_count']>0) | \
                      (repos['created_at']>'2018-01-01T00:00:00Z') | repos['description'].str.contains('paper'))]
    repos = repos.loc[(repos['readme_url'].notnull()) | (repos['forks']>10) | (repos['stargazers_count']>10)]
    # repos = repos.loc[repos['full_name']=='smistad/FAST']

    return repos

def normalize_keywords(x):
    with open('/Users/ianshen/Documents/github/clinical-opensource-projects/data/keyword_maps.json', 'rb') as fp:
        keyword_maps = json.load(fp)
    
    keywords = x[0].split(', ')
    norm_keywords = []
    for keyword in keywords:
        keyword = keyword.strip().lower()
        keyword = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", keyword)
        keyword_map = list(filter(lambda x: x['val']== keyword, keyword_maps))
        if len(keyword_map)>0:
            keyword_eq = keyword_map[0]['eq_val']
            keyword_tags = keyword_map[0]['tag']
            
            if len(keyword_eq) >0:
                norm_keywords = norm_keywords + keyword_eq           
            elif len(keyword_tags) > 0:
                norm_keywords = norm_keywords + keyword_tags[:1]
            else:
                norm_keywords = norm_keywords +[keyword]
        else:
            norm_keywords = norm_keywords +[keyword]

    keywds = ','.join(norm_keywords)

    return keywds

def process_keywords(repos):
    """
    normalize keywords so that they could be used as query terms
    for instance, webbased doctor office ==> doctor office
    """
    repos['norm_keywords'] = repos[['watson_keywords']].apply(lambda x: '' if len(x.values)==0 else normalize_keywords(x.values), axis=1)

    return repos

def get_query_name(x):
    if x[1] in nlp.vocab and x[2]=='Organization' and (x[0] not in x[1] and x[1] not in x[0]):
        return x[0] + ' ' + x[1]
    elif x[0] in x[1]:
        return x[0]
    else:
        return x[1]

def process_fullnames(repos):
    """
    normalize full_names so that they could be used as query terms
    """

    fullnames = repos.full_name
    names = fullnames.str.split('/',  expand=True)
    names.columns = ['username', 'repo_name']
    repos_df = pd.concat([repos, names], axis=1)
    repos_df['repo_name'] = repos_df[['username', 'repo_name', 'owner_type']].apply(lambda x: get_query_name(x.values), axis=1)
    repos_df = repos_df.drop(['username'], axis=1)

    return repos_df

def get_query_terms(repos, type = 'full_name'):
    """
    get list of query terms for each repo, including phrase and start_time
    """

    repos['created_at'] = pd.to_datetime(repos['created_at'])
    if type == "full_name":
        repos['created_year'] = repos.created_at.dt.year
        query_terms = repos[['id','full_name', 'created_year']]
    else:
        keywords = repos.norm_keywords
        keywords = keywords.str.split(',', n=1, expand=True)
        keywords.columns = ['first_keyword','keywords']
        keywords = keywords.fillna('')
        repos_df = pd.concat([repos, keywords], axis=1)
        repos_df['created_year'] = repos_df.created_at.dt.year
        query_terms = repos_df[['id','repo_name', 'first_keyword', 'keywords', 'created_year']]

    return query_terms

def process_arts(config, repo_id, phrase, articles):
    """
    process return articles object, return a list of articles with their basic information
    """
    arts = []
    for art in articles:
        # res = art.as_csv()
        keys = [pair[0] for pair in sorted([(key, val[2]) for key, val in \
                list(art.attrs.items())], key=lambda pair: pair[1])]
        res = [art.attrs[key][0] for key in keys]
        if res[1]: # first check if url is NoneType or not
            if config.SCHOLAR_SITE in res[1]: # the second element is the url of the paper
                res[1] = res[1].replace(config.SCHOLAR_SITE + '/', '')
        res = [repo_id] + [phrase] + res
        arts.append(res)

    return arts

def literature_search(query_terms, type='full_name'):
    """
    perform a google scholar query with given terms
    """

    querier = ScholarQuerier()
    settings = ScholarSettings()
    config = ScholarConf()
    settings.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
    querier.apply_settings(settings)
    query = SearchScholarQuery()

    papers = []
    for item in query_terms.values:
        repo_id = item[0]
        
        if type !='full_name':
            repo_name = item[1]
            phrase = item[2]
            keywords = item[3]
            start_year = item[4]
            if keywords:
                if ',' not in keywords:
                    keywords = keywords + ','
                query.set_words_some(keywords)                

            query.set_words(repo_name)
            query.set_phrase(phrase)

            phrase_text = repo_name + ', ' + phrase
        else:
            phrase = item[1]
            start_year = item[2]

            query.set_phrase(phrase) # commontk/CTK, meoyo/AIPS
            phrase_text = phrase
        print('search papers for {} ...'.format(phrase_text))
        query.set_timeframe(start_year)
        querier.send_query(query)
        articles = querier.articles
        if len(articles)==0:
            continue
        results = process_arts(config, item[0], phrase_text, articles)
        papers = papers + results
        time_delay = random.randrange(1,10)
        time.sleep(time_delay)

    return papers

def main():
    """
    do something
    """
    # repos = load_repos()
    # repos = process_fullnames(repos)
    # repos.loc[repos['watson_keywords'].isnull(), 'watson_keywords'] = ' '
    # repos = process_keywords(repos)

    repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_normalized.csv')
    # repos_0 = pd.read_csv('~/Documents/repos_forks_0.csv')
    # repos = repos.loc[~repos['id'].isin(repos_0.id.values)]
    # print(repos.shape) (2857, 22)

    ranges = [[90*i, 90*(i+1)] for i in range(len(repos)//90)]
    ranges = ranges + [[90*len(ranges), len(repos)]]
    # print(len(ranges)) # 32, 57

    i = 56 # manually choose from 0 to 31 or 0 to 56 for keywords-based search
    rng = ranges[i]
    file = '~/Documents/papers_keywords/papers_' + str(i) + '.csv'
    # print(rng, file)

    query_terms = get_query_terms(repos, type='keywords')
    query_terms = query_terms.iloc[rng[0]:rng[1],]
    # query_terms = query_terms.loc[query_terms['id']==4695314]
    papers = literature_search(query_terms, type='keywords')
    col_names = ['id','full_name','title','url','year','citations','versions','cluster_id', 'url_pdf','url_citations', \
                 'url_version', 'url_citation','excerpt']
    papers_df = pd.DataFrame(papers, columns=col_names)
    # papers_df.head()
    papers_df.to_csv(file, index=False)
    print('\n')

if __name__ == "__main__":
    main()
