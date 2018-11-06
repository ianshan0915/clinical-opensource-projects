# -*- coding=utf-8 -*-
#!/usr/bin/env python3

"""
Created on Wed Aug 27 2018

@author: ianshan0915 (https://github.com/ianshan0915)

Use scholar.py from Christian Kreibich (https://github.com/ckreibich/scholar.py) 
to query papers for each of the extracted clinical/medical related Github repository
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd 


def merge_files(size, dir_path):
  """
  """

  papers_dfs = []
  for i in range(size):
    filenm = 'papers_' + str(i) + '.csv'
    papers = pd.read_csv(dir_path + filenm)
    papers_dfs.append(papers)
  
  papers = pd.concat(papers_dfs)

  return papers

def merge_papers():
  """
  """

  papers_fullname = pd.read_csv('~/Documents/papers_fullname/papers_fullname.csv')
  papers_keywords = pd.read_csv('~/Documents/papers_keywords/papers_keywords.csv')
  papers_fullname = papers_fullname.assign(query_type='fullname')
  papers_keywords = papers_keywords.assign(query_type='keyword')
  papers = pd.concat([papers_fullname, papers_keywords])
  papers.to_csv('/Users/ianshen/Documents/github/clinical-opensource-projects/data/papers.csv', index=False)
  # print(papers.shape) # (9028, 14)

  return None

def main():
  """ do something """

  papers_fullname = merge_files(32, dir_path='~/Documents/papers_fullname/')
  papers_keywords = merge_files(57, dir_path='~/Documents/papers_keywords/')
  papers_fullname.to_csv('~/Documents/papers_fullname/papers_fullname.csv', index=False)
  papers_keywords.to_csv('~/Documents/papers_keywords/papers_keywords.csv', index=False)
  print(papers_fullname.shape, papers_keywords.shape)

if __name__ == "__main__":
  main()