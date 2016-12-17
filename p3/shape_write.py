#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
import update #from update.py
from pymongo import MongoClient

"""
wrangle the data and transform the shape of the data.
The output will be a list of dictionaries that look like this:

{
"id": "2406124091",
"type: "node",
"visible":"true",
"created": {
          "version":"2",
          "changeset":"17206049",
          "timestamp":"2013-08-03T16:43:42Z",
          "user":"linuxUser16",
          "uid":"1219059"
        },
"pos": [41.9757030, -87.6921867],
"address_detail": {
          "housenumber": "5157",
          "postcode": "60625",
          "street": "North Lincoln Ave",
          "city": "上海市"
        },
"name_detail": {
          "zh_pinyin": "Màidāngláo", 
          "en": "McDonald's", 
          "zh": "麦当劳"
        },
"roof_detail": {
          "shape": "gabled",
          "levels": "1"
        },
"building_detail": {
          "levels": "6"
        },
"amenity": "restaurant",
"cuisine": "mexican",
"name": "La Cabana De Don Luis",
"phone": "1 (773)-271-5176"
}

Description of the process:
- process only 2 types of top level tags: "node" and "way"
- all attributes of "node" and "way" are turned into regular key/value pairs, except:
    - attributes in the CREATED array are added under a key "created"
    - attributes for latitude and longitude are added to a "pos" array
- if the second level tag "k" value contains problematic characters, it is ignored
- if the second level tag "k" value starts with "addr:", it is added to a dictionary "address_detail"
- if the second level tag "k" value starts with "name:", it is added to a dictionary "name_detail"
- if the second level tag "k" value starts with "roof:", it is added to a dictionary "roof_detail"
- if the second level tag "k" value starts with "building:", it is added to a dictionary "building_detail"
- if the second level tag "k" value does not start with "addr:" or "roof:" or "building:" or "name:",
  but contains ":", it is added to a dictionary "other_detail"
- for "way" specifically:

    <nd ref="305896090"/>
    <nd ref="1719825889"/>

are turned into
    "node_refs": ["305896090", "1719825889"]
"""


lower = re.compile(r'^([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#Fuction: deal with the second level tag "k" value containing ":", split it into a two-level dictionary 
def process_colon_k(tag,detail,start_str):
    if tag.get("k").startswith(start_str):
        pos_colon = tag.get("k").find(":")
        detail[tag.get("k")[pos_colon+1:]] = tag.get("v") 
        return True
    elif ":" in tag.get("k"):
        detail[tag.get("k")] = tag.get("v") 
        return True
    return False

#Fuction: wrangle and shape one element with top level tag "node" or "way"
def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node["id"] = element.get("id")
        node["type"] = element.tag
        node["visible"] = element.get("visible")
        # transform creation infomation into an array("created")
        created = {}
        for e in CREATED:
            if element.get(e):
                created[e] = element.get(e)
        node["created"] = created
        # process location information ("lon","lat") of "node"
        if element.tag == "node":
            pos = [float(element.get("lat")),float(element.get("lon"))]
            node["pos"] = pos
        # process "refs" of "way"
        else:
            node_refs = []
            for nd in element.iter("nd"):
                node_refs.append(nd.get("ref"))
            if len(node_refs) > 0:
                node["node_refs"] = node_refs
                
        # process the second level tag "k" value 
        address_detail = {}
        name_detail = {}
        roof_detail = {}
        building_detail = {}
        other_detail = {}
        for tag in element.iter("tag"):
            #second level tag "k" value contains problem characters
            if problemchars.match(tag.get("k")): 
                pass
            # second level tag "k" value without ":"
            elif lower.match(tag.get("k")):
                node[tag.get("k")] = tag.get("v")
            # second level tag "k" value with ":", add it to the specified dictionary
            else:
                if  tag.get("k").startswith("addr:"):
                    #update city information using fuction "update_city" from "update.py"
                    if tag.get('k') == "addr:city":
                        address_detail["city"] =  update.update_city(tag.get("v"))
                    #update street information using fuction "update_street" from "update.py"
                    elif  tag.get('k') == "addr:street":
                        address_detail["street"] =  update.update_street(tag.get("v"))
                    else:
                        address_detail[tag.get("k")[tag.get("k").find(":")+1:]] = tag.get("v")
                elif process_colon_k(tag,name_detail,"name:"):
                     pass
                elif process_colon_k(tag,roof_detail,"roof:"):
                    pass
                elif process_colon_k(tag,building_detail,"building:"):
                    pass
                elif process_colon_k(tag,other_detail,":"):
                    pass


        if address_detail:
            node["address_detail"] = address_detail
        if name_detail:
            node["name_detail"] = name_detail
        if roof_detail:
            node["roof_detail"] = roof_detail
        if building_detail:
            node["building_detail"] = building_detail  
        if other_detail:
            node["other_detail"] = other_detail
    return node


#Fuction:wrangle and shape elements, return a list processed elements, write out into a json file
def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data



if __name__ == '__main__':
    sample_data = process_map("sample.osm")

    #insert sample data into MongoDB
    client = MongoClient('localhost:27017')
    db = client['DAND']
    db.test.drop()
    db.test.insert_many(sample_data) 
    print "sample data has {0} documents.".format(db.test.find().count())





