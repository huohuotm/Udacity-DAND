#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import xml.etree.cElementTree as ET
import pprint

"""look into street types
and write out street names of different types"""


OSMFILE = "sample.osm"
output_file = 'street_detail.txt'

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

#Fuction: add one street name to street type 
def audit_street_type(street_types,street_name):
    status = 0
    for street_type in ['弄','路','道','街','Road','Street','Avenue']:
        if street_name.find(street_type.decode("utf-8")) != -1:
            street_types[street_type.decode("utf-8")].add(street_name)
            status = 1
    if not status:
        street_types['others'].add(street_name)

#Fuction: add all street names to street types respectively
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("end",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag) :
                    audit_street_type(street_types,tag.attrib['v'])                   
    osm_file.close()
    return street_types

# write out street names of different street types
if __name__ == "__main__":
    street_types = audit(OSMFILE)
    with open(output_file, 'wb') as output:
        for street_type in ['弄','路','道','街','Road','Street','Avenue','others']:
            roads = street_types[street_type.decode('utf-8')]
            output.write("-----------------------------------\n")
            output.write("This section is street names of " + street_type + ":\n\n")
            for road in roads:
                line = road + "\n"
                output.write(line.encode('utf-8')) 



