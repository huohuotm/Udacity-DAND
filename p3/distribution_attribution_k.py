#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
from collections import defaultdict

""" calculate distribution of subtag's attribution_k 
and write out sorted by frequency in descending order"""

input_file = "sample.osm"
output_file = "attribution_k_distribution.txt"
 
# Fuction: calculate distribution of subtag's attribution_k, return a dictionary {attribution_k:frequency}  
def attribution_counts(filename, tags=('node', 'way')):
    keys = defaultdict(int)
    for _, element in ET.iterparse(filename, events =('end',)):
        if element.tag in tags and element.iter('tag'):
            for ele in element.iter('tag'):
                keys[ele.get('k')] += 1
    return keys

#Fuction: return keys of a dictionary sorted by value in descending order
def order_desc(keys):
    keys_dic = dict(keys)
    keys_list = list(keys)
    return sorted(keys_list,reverse=True,key=lambda k: keys_dic[k])


#Fuction: write out distribution of attribution_k
def process_map(datafile):
    keys = attribution_counts(datafile)
    soretd_keys = order_desc(keys)
    with open(output_file, 'wb') as output:
        for key in soretd_keys:
            line = "attribution_k: '"+ key + "' appears " + str(keys[key]) + " times\n"
            output.write(line)


if __name__ == "__main__":
    process_map(input_file)