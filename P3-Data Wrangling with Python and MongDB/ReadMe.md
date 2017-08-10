# OpenStreetMap Project  
Data Wrangling with MongoDB

### Documents:

* `DataWrangle_OSM ` document data wrangling process
* `sample.osm`
* `mapping.text`    used in `update.py` to update city name


### Code:
* `get_sample.py`    take a sample of elements from original OSM region.[refer](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345404/modules/316820862075463/lessons/3168208620239847/concepts/77135319070923#)
* `distribution_attribution_k.py` get distribution of attributions
* `look_into_city.py`    explore city names
* `look_into_street.py`    explore street names
* `update.py`            function to update city name and street name
* `shape_write.py`        wrangling process



##Notice:
Run `update.py` before `shape_write.py`, because the latter import "update" as a new library.