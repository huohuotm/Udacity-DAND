#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import xml.etree.cElementTree as ET
import pprint

"""look into city types
and write out city names of different types"""


OSMFILE = "sample.osm"
output_file = 'city_detail.txt'

def is_city_name(elem):
    return (elem.attrib['k'] == "addr:city")

#Fuction: add one city name to city type 
def audit_city_type(city_types,city_name):
    status = 0
    for city_type in ['市','区','City','District']:
        if city_name.find(city_type.decode("utf-8")) != -1:
            city_types[city_type.decode("utf-8")].add(city_name)
            status = 1
    if not status:
        city_types['others'].add(city_name)

#Fuction: add all city names to city types respectively
def audit(osmfile):
    osm_file = open(osmfile, "r")
    city_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("end",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_city_name(tag) :
                    audit_city_type(city_types,tag.attrib['v'])                   
    osm_file.close()
    return city_types

# write out city names of different city types
if __name__ == "__main__":
    city_types = audit(OSMFILE)
    with open(output_file, 'wb') as output:
        for city_type in ['市','区','City','District','others']:
            roads = city_types[city_type.decode('utf-8')]
            output.write("-----------------------------------\n")
            output.write("This section is city names of " + city_type + ":\n\n")
            for road in roads:
                line = road + "\n"
                output.write(line.encode('utf-8')) 



