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

import pandas as pd

def load_repos():
    """
    load the repository and do some basic filtering
    """

    repos = pd.read_csv('')

    return repos

def test():
    """
    perform a test query with a given phrase
    """

    querier = ScholarQuerier()
    settings = ScholarSettings()
    settings.set_citation_format(ScholarSettings.CITFORM_BIBTEX)
    querier.apply_settings(settings)

    query = SearchScholarQuery()
    query.set_phrase('commontk/CTK') # commontk/CTK, meoyo/AIPS
    query.set_timeframe('2008')
    querier.send_query(query)
    articles = querier.articles

    return articles

def main():
    """
    do something
    """
    config = ScholarConf()
    articles = test()
    for art in articles:
        # res = art.as_csv()
        keys = [pair[0] for pair in sorted([(key, val[2]) for key, val in \
                list(art.attrs.items())], key=lambda pair: pair[1])]
        res = [art.attrs[key][0] for key in keys]
        if config.SCHOLAR_SITE in res[1]: # the second element is the url of the paper
            res[1] = res[1].replace(config.SCHOLAR_SITE, '')
        print(res)

if __name__ == "__main__":
    main()
