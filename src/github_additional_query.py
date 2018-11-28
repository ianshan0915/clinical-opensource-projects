# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed August 04 2018

@author: ianshan0915

Use GitHub REST API v3 to query open sourced repos related to clinical/medical
"""

import os
import requests
import json

import pandas as pd
import numpy as np

def fetch_topics(url):
  """
  fetch topics given a repo url
  """

  token = os.environ['GithubToken']
  auth = 'token ' + token
  headers = {'Accept': 'application/vnd.github.mercy-preview+json', 'Authorization': auth}
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

def fetch_readme_size(url):
  """
  fetch size of the readme file of a given a repo url
  """

  print(url)
  token = os.environ['GithubToken']
  auth = 'token ' + token
  headers = {'Authorization': auth}
  response = requests.request("GET", url, headers=headers)
  resp_json = response.json()
  if 'size' in resp_json.keys():
    size = resp_json['size']
  else:
    size = -1
  print(size)
  return size

def gen_readme_size():
  """
  get readme file size for github repositories given their api url
  """

  repos = pd.read_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_topics.csv')
  repos.loc[repos['readme_url'].isnull(),'readme_url'] = ''
  dfs = np.array_split(repos, 3)
  # repos = pd.read_csv('/Users/ianshen/Documents/readme_url_issues.csv')
  # dfs[0]['readme_size'] = dfs[0].readme_url.apply(lambda x: 0 if x=='' else fetch_readme_size(x))
  # dfs[0].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_readmesize_1.csv', index=False)
  # dfs[1]['readme_size'] = dfs[1].readme_url.apply(lambda x: 0 if x=='' else fetch_readme_size(x))
  # dfs[1].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_readmesize_2.csv', index=False)
  dfs[2]['readme_size'] = dfs[2].readme_url.apply(lambda x: 0 if x=='' else fetch_readme_size(x))
  dfs[2].to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_readmesize_3.csv', index=False)

  # repos['readme_size'] = repos.readme_url.apply(lambda x: 0 if x=='' else fetch_readme_size(x))
  # repos.to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/repos_readmesize_3.csv', index=False)

  return None

def main():
  # do something

  gen_readme_size()

if __name__ == '__main__':
  main()
