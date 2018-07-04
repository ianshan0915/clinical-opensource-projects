# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed May 30 2018

@author: ianshan0915

Use GitHub REST API v3 to query open sourced repos related to clinical/medical
"""

import pandas as pd

def load_data():
  """
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos.csv')

  # preprocess the repos' language column
  repos.loc[repos['language'].isin(['CSS','HTML']), 'language'] = 'CSS/HTML'
  repos.loc[repos['language'].isin(['TypeScript']), 'language'] = 'JavaScript'
  repos = repos.loc[repos.language.notnull()] # repos with no language(s) indicated are removed

  return repos

def check_lang_trends(df):
  """
  to uncover the evolution of progoramming languages  
  """

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

def others():
  """
  """

  pass

  return None

def main():
  """do a list of tasks"""

  repos = load_data()
  langs_trends = check_lang_trends(repos)

if __name__ == "__main__":
  main()