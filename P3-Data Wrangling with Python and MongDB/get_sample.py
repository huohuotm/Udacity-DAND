#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET  

"""extract sample.osm file from original file(shanghai_china.osm)
and write out sample.osm"""


OSM_FILE = "/Users/weidian1/Documents/Study/nanodegreee/P3/shanghai_china.osm"  
SAMPLE_FILE = "sample.osm"

k = 100 # Parameter: take every k-th top level element


#Fuction: extract sample.osm file
def get_element(osm_file, tags=('node', 'way', 'relation')):
    context = iter(ET.iterparse(osm_file, events=('start', 'end'))) 
    _, root = next(context)   
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()     

#Fuction: write out sample.osm
def process_map(datafile):
    with open(SAMPLE_FILE, 'wb') as output:
        output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write('<osm>\n  ')
        # Write every kth top level element
        for i, element in enumerate(get_element(datafile)):
            if i % k == 0:
                output.write(ET.tostring(element, encoding='utf-8'))
        output.write('</osm>')
    

if __name__ == '__main__':
    process_map(OSM_FILE)