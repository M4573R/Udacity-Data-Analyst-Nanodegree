#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pprint
import re
import codecs
import json

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
zipcode_re = re.compile(r'[^0-9\-]', re.IGNORECASE)
state_OR_re = re.compile(r'.+Portland.+|Or.*|.+Street|97222|oregon')
state_WA_re = re.compile(r'Wa.*')
city_portland_re = re.compile(r'[Pp]ortland') 
city_other_re = re.compile(r',\sOR$')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

expected = ["Avenue", "Boulevard", "Broadway", "Circle", "Court", "Drive", "Highway", "Lane", "Parkway", "Place", 
            "Road", "Street", "Terrace", "Way"]

mapping = { 'St'     : 'Street',
            'St.'    : 'Street',
            'street' : 'Street',
            'Street,': 'Street',
            'Rd'     : "Road",
            'Rd.'    : 'Road',
            'ROAD'   : 'Road',
            'road'   : 'Road',
            'Pkwy'   : 'Parkway',
            'Pky'    : 'Parkway',
            'Ln'     : 'Lane',
            'Hwy'    : 'Highway',
            'Dr'     : 'Drive',
            'Dr.'    : 'Drive',
            'Cir'    : 'Circle',
            'Blvd'   : 'Boulevard',
            'Blvd.'  : 'Boulevard',
            'ave'    : 'Avenue',
            'Ave'    : 'Avenue',
            'Ave.'   : 'Avenue' }

def shape_element(element):
    if element.tag == "node" or element.tag == "way" :
        node = {}
        node["created"] = {}
        node["type"] = element.tag
        for babytree in element.getchildren():
            if babytree.tag == "tag":
                #if "address" not in node.keys():
                    #node['address'] = {}    
                if re.search("^addr\:.+$",babytree.attrib['k']):
                    if "address" not in node.keys():
                        node['address'] = {}
                    match = re.search("\w+$",babytree.attrib['k'])
                    field = match.group()
                    if field == 'street':
                        street_name = babytree.attrib['v']
                        m1 = street_type_re.search(street_name)
                        if m1:
                            street_type = m1.group() #The 'word' of the street_name
                            if street_type not in expected and street_type in mapping.keys():
                                street_name = re.sub(street_type, mapping[street_type], street_name)
                        node["address"][field] = street_name
                    elif field == 'postcode':
                        zipcode = babytree.attrib['v']
                        m2 = zipcode_re.search(zipcode)
                        if m2:
                            #Remove any character(or space) other than number and "-"
                            zipcode = re.sub(r'([^0-9\-])', '', zipcode) 
                        node["address"][field] = zipcode
                    elif field == 'state':
                        state = babytree.attrib['v']
                        m31 = state_OR_re.search(state)
                        m32 = state_WA_re.search(state)
                        if m31:
                            # Fix the weird state name that should be 'OR'
                            state = re.sub(r'.+Portland.+|Or.*|.+Street|97222|oregon', 'OR', state)
                            node["address"][field] = state
                        elif m32:
                            # Fix the weird state name that should be 'WA'
                            state = re.sub(r'Wa.*', 'WA', state)
                        node["address"][field] = state
                    elif field == 'city':
                        city = babytree.attrib['v']
                        m41 = city_portland_re.search(city)
                        m42 = city_other_re.search(city)
                        if m41: #Search if thers's 'portland' or 'Portland' in the string
                            node["address"][field] = 'Portland'
                        elif m42:
                            # If the city name is ended with ', OR', then remove it
                            city = re.sub(r',\sOR$', '', city)
                        node["address"][field] = city
                        
                    else:
                        node["address"][field] = babytree.attrib['v']

                else: 
                    node[babytree.attrib['k']] = babytree.attrib['v']

        for attrib in element.attrib:
            if attrib not in CREATED:
                node[attrib] = element.attrib[attrib]
            else:
                node["created"][attrib] = element.attrib[attrib]
        if element.tag == "node":
            node["pos"] = [element.attrib["lat"], element.attrib["lon"]]
        else:
            node["node_refs"] = []
            for tag in element.iter("nd"):
                node["node_refs"].append(tag.attrib['ref'])
        return node
    else:
        return None

    
    
def process_map(file_in, pretty = False):
    # You do not need to change this file
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

OSMFILE = '~/Documents/Open_Course/UDACITY/Data Wrangling with MongoDB/Final Project/portland_oregon.osm'
data = process_map(OSMFILE, False)