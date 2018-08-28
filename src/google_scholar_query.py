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

import pandas as pd

def load_repos():
    """
    load the repository and do some basic filtering
    """

    repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
    repos = repos.loc[repos['forks']>0]
    # repos = repos.loc[repos['full_name']=='smistad/FAST']

    return repos

def get_query_terms(repos):
    """
    get list of query terms for each repo, including phrase and start_time
    """

    repos['created_at'] = pd.to_datetime(repos['created_at'])
    repos['created_year'] = repos.created_at.dt.year
    query_terms = repos[['id','full_name', 'created_year']]

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

def literature_search(query_terms):
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
        phrase = item[1]
        start_year = item[2]
        query.set_phrase(phrase) # commontk/CTK, meoyo/AIPS
        query.set_timeframe(start_year)
        print('search papers for {} ...'.format(phrase))
        querier.send_query(query)
        articles = querier.articles
        if len(articles)==0:
            continue
        results = process_arts(config, item[0], phrase, articles)
        papers = papers + results
        time.sleep(1)

    return papers

def main():
    """
    do something
    """
    repos = load_repos()
    query_terms = get_query_terms(repos)
    query_terms = query_terms.head(100)
    papers = literature_search(query_terms)
    col_names = ['id','full_name','title','url','year','citations','versions','cluster_id', 'url_pdf','url_citations', \
                 'url_version', 'url_citation','excerpt']
    papers_df = pd.DataFrame(papers, columns=col_names)
    # papers_df.head()
    papers_df.to_csv('~/Documents/papers_test.csv', index=False)

if __name__ == "__main__":
    main()
