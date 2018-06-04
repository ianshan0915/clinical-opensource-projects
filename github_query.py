# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed May 30 2018

@author: ianshan0915

Use GitHub REST API v3 to query open sourced repos related to clinical/medical
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import time
import os
from datetime import datetime
from datetime import timedelta

import pandas as pd
import pymysql as mysql
from sqlalchemy import create_engine

from github import Github

def github_conn(token):
  """
  set up a connection with github api
  """

  g = Github(token)

  return g

def search_timerange_year(type, year):
  """
  get search timerange on a given year
  """
  if type==1:
    timerange = '<' + str(year) + '-01-01'
  else:
    start = str(year) + '-01-01'
    if year == datetime.now().year:
      end = datetime.now().strftime('%Y-%m-%d')
    else:
      end = str(year) + '-12-31'
    timerange = start + '..' + end
  
  return timerange

def split_time(timerange, splitsize):
  """
  split timerange into multiple timeranges so that each timerange would only contains less than 1000 repos
  """

  times = timerange.split('..')
  start = times[0]
  end = times[1]
  time_format = '%Y-%m-%d'
  timeranges_year = []
  duration_days = (datetime.strptime(end, time_format)-datetime.strptime(start, time_format)).days
  delta_days = duration_days//splitsize
  for i in range(splitsize):
    if i< splitsize-1:
      end_tmp = datetime.strptime(start, time_format) + timedelta(days=delta_days)
      timerange_tmp = start + '..' + end_tmp.strftime(time_format)
      start_tmp = end_tmp + timedelta(days=1)
      start = start_tmp.strftime(time_format)
    else:
      timerange_tmp = start + '..' + end

    timeranges_year.append(timerange_tmp)

  return timeranges_year

def get_timeranges(conn, terms, cutoff, curr):
  """
  generate timerange list that will limit repositories returned to 1000
  a cutoff year is obtained after trying the search term on github.com, repos before this cutoff year is less 1000
  """
  timeranges = []
  timerange_pre = search_timerange_year(1, cutoff)

  ## double-check if the cutoff year is good or not
  pagedlist_pre = conn.search_repositories(query = terms, created=timerange_pre)
  if pagedlist_pre.totalCount>1000:
    print('the cutoff year is not correct, should choose a time before it')
  else:
    timeranges.append(timerange_pre)

  while cutoff <=curr:
    timerange = search_timerange_year(2, cutoff)
    # print(timerange)
    pagedlist_tmp = conn.search_repositories(query = terms, created=timerange)
    splitsize = math.ceil(pagedlist_tmp.totalCount/1000) + 2
    timerange_lst = split_time(timerange, splitsize)
    timeranges = timeranges + timerange_lst
    cutoff+=1

  return timeranges

def fetch_content(repo, contentName):
  """
  check if a repo has a given content
  """

  # print(repo)
  url = repo.url + '/' + contentName
  status = repo._requester.requestJson('GET', url)[0]
  if status ==200:
    return repo.get_readme().url
  else:
    return None

def repos_query(conn, terms, tmranges, attrs, remaining_rate):
  """
  obtain repos list based on query terms
  """

  ## get repositories for each timerange
  repos = []
  for timerange in tmranges:
    print("timerange is ",timerange)
    print("Remaining rates are ", remaining_rate)
    paged_list = conn.search_repositories(query = terms, created=timerange)
    if paged_list.totalCount >1000:
      print('too many repos in this time range (', timerange, ')')
      break
    elif remaining_rate < paged_list.totalCount*2:
      print("no left rates, how to break")
      conn = github_conn(os.environ['GithubToken'])
      sleep_dur = get_sleep_time(conn)
      time.sleep(sleep_dur)
      paged_list = conn.search_repositories(query = terms, created=timerange)
      repos_tmp = [[item._rawData[k] for k in attrs] + [fetch_content(item, 'readme')] for item in paged_list]
      repos = repos + repos_tmp
    else:
      # print(paged_list.totalCount)
      repos_tmp = [[item._rawData[k] for k in attrs] + [fetch_content(item, 'readme')] for item in paged_list]
      repos = repos + repos_tmp
      time.sleep(120)
    conn = github_conn(os.environ['GithubToken'])
    remaining_rate = conn.rate_limiting[0]

  return repos

def mysql_insert(df):
  """
  insert dataframe into mysql db in gcloud
  """

  db_conn = os.environ['DbSetting']
  engine = create_engine(db_conn)
  df.to_sql('clinical_github', engine, if_exists='append', index=False)
  engine.dispose()

def get_sleep_time(conn):
  """
  Calculate the amount of seconds it needs to wait before remaining rate reset
  """

  tdelta = datetime.fromtimestamp(conn.rate_limiting_resettime) - datetime.now()

  return tdelta.seconds +30 ## 30 seconds after the remaining rate reset


def check_remain_rates(conn):
  """
  check whether there if any remaining rates
  """

  remaining = conn.rate_limiting[0]
  print("left rates: ",remaining)
  if remaining ==0:
    print('No remaining rates left!')
    return False
  else:
    return True

def process(conn, terms, attrs):
  """
  a series of actions to split the big query and conduct divided small queries
  """

  # divide big query into smaller ones with less than 1000 repos
  tmranges = get_timeranges(conn, terms, 2013, 2018)
  # debugging print
  for tmrange in tmranges:
    print(tmrange)

  conn = github_conn(os.environ['GithubToken'])
  remaining_rate = conn.rate_limiting[0]
  print("remaining rate after split big query: ", remaining_rate)
  # conduct small queries
  repos = repos_query(conn, terms, tmranges, attrs, remaining_rate)
  print(len(repos))
  # remove the duplicates and insert into mysql db
  df_repos = pd.DataFrame(repos, columns=attrs+['readme_url'])
  df = df_repos.drop_duplicates(subset=['id'])
  df_duplicates = df_repos.loc[~df_repos.index.isin(df.index)]
  df_duplicates.to_csv('/Users/ianshen/Documents/repos_duplicates.csv', index=False)
  df.to_csv('/Users/ianshen/Documents/repos.csv', index=False)
  # mysql_insert(df)
  # print('one insertion is completed!')

def main():
  """
  query the repos based a given query statement
  """
  
  # set up configurations
  token = os.environ['GithubToken']
  terms = '(clinical OR medical) OR (patient OR doctor)'
  conn = github_conn(token)
  attrs = ['id','full_name','url','description','created_at', 'updated_at', \
          'pushed_at', 'forks', 'stargazers_count','language']

  # check remaining rates before start
  if check_remain_rates(conn):
    print('we have rates!')  
    process(conn, terms, attrs)
  else:
    sleep_dur = get_sleep_time(conn)
    print('sleep ', sleep_dur)
    time.sleep(sleep_dur)

    # double check if the rate is larger than 0
    try:
      # process(conn, terms, attrs)
      pass
    except Exception as e:
      raise e

if __name__ == '__main__':
  main()
    