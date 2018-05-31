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
    print(timerange)
    pagedlist_tmp = conn.search_repositories(query = terms, created=timerange)
    splitsize = math.ceil(pagedlist_tmp.totalCount/1000)
    timerange_lst = split_time(timerange, splitsize)
    timeranges = timeranges + timerange_lst
    cutoff+=1

  return timeranges

def repos_query(conn, terms, tmranges):
  """
  obtain repos list based on query terms
  """

  ## get repositories for each timerange
  repos = []
  for timerange in tmranges:
    paged_list = conn.search_repositories(query = terms, created=timerange)
    if paged_list.totalCount >1000:
      print('too many repos in this time range (', timerange, ')')
      break
    else:
      print(paged_list.totalCount)
      repos_tmp = [item for item in paged_list]
      repos = repos + repos_tmp
      time.sleep(10)

  return repos

def main():
  """
  query the repos based a given query statement
  """
  
  token = os.environ['GithubToken']
  terms = '(clinical OR medical) OR (patient OR doctor)'
  conn = github_conn(token)
  repos = repos_query(conn, terms, ['2013-01-01..2013-12-31'])
  # tmranges = get_timeranges(conn, terms, 2013, 2018)
  print(len(repos))

if __name__ == '__main__':
  main()
    