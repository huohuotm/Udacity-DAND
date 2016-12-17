#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import xml.etree.cElementTree as ET
import ast
import re
"""update streets name
Description of the process:
- if a street has both Chinese and English name("世纪大道 Century Avenue"), keep Chinese part("世纪大道")
- if street name over-abbreviated or inconsistent, change it to official form, which looks like the following:
    -change "street" to "Street"
    -change "avenue" to "Ave."
    -change these "Road" alias('Rd','Rd.','Rd,','rd','Rode','Raod','lu','Lu') to "Road"
- if street name includes other redundant information like house number, place name or meaningless characters, get rid of these redundant information
    - For Chinese street name, cut out information after last key word("路",道","街","街道")
    - For English street name, use method of regular ecpression
"""


#Fuction: remove number and character
def drop_number_character(street_name):
    p = re.compile(r"\w*",re.I)
    return p.sub("",street_name)

#Fuction: return content before last key-word inorder to remove redundant information
def get_content_before(street_name,key_word):
    pos = street_name[::-1].find(key_word)
    return street_name[::-1][pos:][::-1]

#Fuction: update street name of different street types
def update_street(street_name):
    if '路'.decode("utf-8") in street_name:
        street_name = get_content_before(street_name,'路'.decode('utf-8'))
        return drop_number_character(street_name)
    if '道'.decode("utf-8") in street_name:
        return get_content_before(street_name,'道'.decode('utf-8'))    
    if '街'.decode("utf-8") in street_name and '道'.decode("utf-8") not in street_name: #in case of encountering street type of "xxx街道"
        street_name = get_content_before(street_name,'街'.decode('utf-8'))
        return drop_number_character(street_name)
    # update Street alias to Street
    if 'street' in street_name:
        return street_name.replace('street','Street')
    # update Road alias to Road
    for Road_alias in ['Rd','Rd.','Rd,','rd','Rode','Raod','lu','Lu']:
         if Road_alias in street_name :
            street_name = street_name.replace(Road_alias,'Road')   
    # update Road, remove redundant information, characters and numbers
    if 'Road' in street_name:
        street_name = get_content_before(street_name,'Road'[::-1])       #return content before last "Road" to remove redundant information
        m = re.search(r"(Lane|Alley)+",street_name)
        n = re.search(r"[\d#-]+",street_name)                            # don't match ",", because I don't want change "xx Road, aa Road" to "xx Road aa Road"
        if m:
            street_name = re.sub(r"(Lane|Alley)","",street_name).strip() #remove "Lane","Alley",numbers, "-","#","N0."
        if n:
            street_name = re.sub(r"[\d#-,]","",street_name).strip()      #remove  "-","#","N0.",","
        return street_name
    # update Avenue alias to Avenue
    for Avenue_alias in ['avenue','Ave.']:
         if Avenue_alias in street_name:
            return street_name.replace(Avenue_alias,'Avenue')   
    
    return street_name



"""update city name
Load a file("mapping.text") to mapping variable as a dictionary in format of {original city name : updated city name}),
then, update the city name that matches key value
"""

def is_city_name(elem):
    return (elem.attrib['k'] == "addr:city")


# "mapping.text" is a dictionary in format of {original city name : updated city name}
with open('mapping.text','r') as m:
    mapping = ast.literal_eval(m.read()) #convert str to dict


# Fuction: update city name by "mapping" dictionary
def update_city(city_name):
    if city_name.encode('utf-8') in mapping.keys():
        return mapping[city_name.encode('utf-8')]
    return city_name
        


