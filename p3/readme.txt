OpenStreetMap Project 
Data Wrangling with MongoDB

Documents:

- "DataWrangle_OSM "			 document data wrangling process
- "sample.osm"
- "mapping.text"    			 used in "update.py" to update city name


Code:
- "get_sample.py"   			 take a sample of elements from original OSM region, refer to code in lesson
- "distribution_attribution_k.py"	 get distribution of attributions
- "look_into_city.py"   		 explore city names
- "look_into_street.py" 		 explore street names
- "update.py"            		 function to update city name and street name
- "shape_write.py"        		 wrangling process


Reason for this area:
I’m a Chinese student, and I like Shanghai.


Notice:
Run "update.py" before "shape_write.py", because the latter import "update" as a new library.
