---
typora-copy-images-to: ./D3js_pic
---

## Data Visualization in Data Analysis

![6945BED0-7786-4D86-B096-6EEB5270512A](D3js_pic/6945BED0-7786-4D86-B096-6EEB5270512A.png)

* Quatitative Data
* Categorical Data
  * Nominal (like geographical region)
  * Ordered (popualtion bins, class difficulty)

#### Visual Encodings:

1. planar variables
2. retinal variable( like ,size, orientation, color saturation  ) ![934F816D-5D0B-42F7-B838-9F8850FA8268](D3js_pic/934F816D-5D0B-42F7-B838-9F8850FA8268.png)

#### 识别度

![B75D9FD9-7680-4B52-BA4E-14803DF8A594](D3js_pic/B75D9FD9-7680-4B52-BA4E-14803DF8A594.png)



## Up and Down the Visualization Spectrum

![BD7CB4E2-C9DA-4FCA-BF90-D0DCE8BBDF96](D3js_pic/BD7CB4E2-C9DA-4FCA-BF90-D0DCE8BBDF96.png)

![B888B42C-E5AB-4469-8E62-BC3F5E2D1A53](D3js_pic/B888B42C-E5AB-4469-8E62-BC3F5E2D1A53.png)

#### DOM

![90E9DC5A-85B0-4AF4-A9E7-D50EF7EC4396](D3js_pic/90E9DC5A-85B0-4AF4-A9E7-D50EF7EC4396.png)

**DOM** (Document Object Model) is a specification, much like html, css, that specifies a common programming interface for html and xml documents.

Once the source of the html is returned from the server, the web browser parses the file and transforms it into a hierarchical object that can accessed programmatically, most often through JavaScript, called the DOM.

D3 binds data to the DOM rather than the source or visual element of the html.

![0758DC3A-5C2E-4B40-8EA6-5B6F82F54A78](D3js_pic/0758DC3A-5C2E-4B40-8EA6-5B6F82F54A78.png)



## D3 Module

#### D3 Syntax

##### Select jquery

1. get element by id![B4C2B849-23D4-4D89-B499-F13676C8062B](D3js_pic/B4C2B849-23D4-4D89-B499-F13676C8062B.png)

   `document.getElementById("footer");` 

2. query selector(css systax)

   ![070ED078-C890-4EA2-90D9-9C8C3F98FE8F](D3js_pic/070ED078-C890-4EA2-90D9-9C8C3F98FE8F.png)



##### 使用d3js模块

**引入d3js**

1. 直接添加  

```html
<!DOCTYPE html>
<html>
<head>
  	<script src="https://d3js.org/d3.v4.min.js"></script>
	<!--或者<script type="text/javascript" src="d3.v2.min.js"></script>-->
</head>
<body> 
```

2. wirte js code in console

```javascript
var script = document.createElement('script')
script.type = 'text/javascript'
script.src = 'https://d3js.org/d3.v3.min.js';
document.head.appendChild(script);
```
![EE932FBA-3581-4521-BC26-778A4D76F292](D3js_pic/EE932FBA-3581-4521-BC26-778A4D76F292.png)

**Usage Example**

`var elem = d3.select(".main");`  elem 是一个DOM node.

```javascript
d3.select("element_name") //return first element named "element_name"
d3.selectAll("img") // return all "img"
d3.select(".class_name")  
d3.select("#id_name")

//两级class <a class="navbar-brand logo" id="12345" >
d3.select(".navbar-brand.logo")

//select id of log, then select child img, change logo
d3.select("#header-logo img").attr("alt", "Udacity");

//置空
d3.select(".main").html(null); 
d3.select(".main").html(""); 
```

##### Chain rule

`var elem = d3.select(".navbar");`

`elem.style("background-color", "gray");` 

等价于

`d3.select(".navbar").style("background-color", "gray");` 



#### Map

map data value—>pixel value

![11EDF9CA-536C-413F-BDE1-5CF66A0E4D6C](D3js_pic/11EDF9CA-536C-413F-BDE1-5CF66A0E4D6C.png)

SVG 左上角坐标（0，0）。

从 15-90 缩放到 250-0，返回map映射函数。

`var x=d3.scale.log().domain([250,100000]).range([0,600]);`

##### Usage of Creating Axes

![99D26829-DF48-422D-8FDD-721D6AC5ED24](D3js_pic/99D26829-DF48-422D-8FDD-721D6AC5ED24.png)

#### Data Join

[ref](https://bost.ocks.org/mike/join/) 

![92DF9D51-1359-4EF8-84C0-A3E3CAFE3B4B](D3js_pic/92DF9D51-1359-4EF8-84C0-A3E3CAFE3B4B.png)

![D60FAF32-D759-4D14-A38E-B01F79ABF38D](D3js_pic/D60FAF32-D759-4D14-A38E-B01F79ABF38D.png)

##### Load Data

```javascript
d3.tsv(url, row, callback);
// is equivalent to this
d3.tsv(url)
  .row(row)
  .get(callbacl);
```

##### Example

![0B764F55-5AA0-4DAE-B96C-02B76D5E2B45](D3js_pic/0B764F55-5AA0-4DAE-B96C-02B76D5E2B45.png)

![451EB824-4A7B-4B92-9243-6AD8B935F106](../../../../../../var/folders/5d/905ndm6s3738zfg7byy8dhhm0000gn/T/abnerworks.Typora/FF1548FE-ACED-4798-9644-12B73B140982.png)

![D2726B6E-6BDB-438D-B2EC-7C3DC5151FF2](D3js_pic/D2726B6E-6BDB-438D-B2EC-7C3DC5151FF2.png)

![D13A1133-86C7-4EA6-963B-B63244A385C7](D3js_pic/D13A1133-86C7-4EA6-963B-B63244A385C7.png)

**empty placeholder nodes**, something like a virtual html node that exists in javascript scope or in the console but is not visible on the page, as an SVG element.



blue：row of data.tsv which are not bound to html/svg elements currently on the page.

purple: row of data.tsv which are bound to html/svg elements currently on the page.

red: html/svg elements currently on the page which are not bound to  row of data.tsv.

![E052EEE2-DDA9-4DA7-8F56-EE3B32A561F5](D3js_pic/E052EEE2-DDA9-4DA7-8F56-EE3B32A561F5.png)

![A9EE5BFD-E8D6-452D-9FE1-F4E603F97CF1](D3js_pic/A9EE5BFD-E8D6-452D-9FE1-F4E603F97CF1.png)

**refs:**

https://bost.ocks.org/mike/circles/

https://bost.ocks.org/mike/join/

http://alignedleft.com/tutorials/d3/binding-data





## Design Principle

![6A26A627-C190-4024-84B3-A297208777CE](D3js_pic/6A26A627-C190-4024-84B3-A297208777CE.png)

[ref-colorbrewer](http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3)

[ref-the-gestalt-laws-of-perception](https://www.slideshare.net/luisaepv/the-gestalt-laws-of-perception) 



## Grammar of Graphics

![A50BD428-A177-4224-B100-F96E9C8756E8](D3js_pic/A50BD428-A177-4224-B100-F96E9C8756E8.png)

![C27895D9-D1BF-43B0-8E3E-6934F0D15662](D3js_pic/C27895D9-D1BF-43B0-8E3E-6934F0D15662.png)

![56AC120E-4EF3-4D2F-954B-E5F2B4B05D6F](D3js_pic/56AC120E-4EF3-4D2F-954B-E5F2B4B05D6F.png)



## Dimple Module

![6A17BA61-655E-4924-A63F-1A7D6CD4E0C7](D3js_pic/6A17BA61-655E-4924-A63F-1A7D6CD4E0C7.png)



##### example

```javascript
// 折线+散点图
// 添加坐标轴，设置时间格式
var myChart = new dimple.chart(svg, data);
var x = myChart.addTimeAxis("x", "year"); 
myChart.addMeasureAxis("y", "attendance");
x.dateParseFormat = "%Y";
x.tickFormat = "%Y";
x.timeInterval = 4;
myChart.addSeries(null, dimple.plot.line);
myChart.addSeries(null, dimple.plot.scatter);
myChart.draw();
```

```javascript
// 气泡图
svg.append('g')
   .attr("class", "bubble")
   .selectAll("circle")
   .data(nested.sort(function(a, b) { 
      return b.values['attendance'] - a.values['attendance'];
   }), key_func)
   .enter()
   .append("circle")
// 气泡的位置及半径
   .attr('cx', function(d) { return d.values['x']; })
   .attr('cy', function(d) { return d.values['y']; })
   .attr('r', function(d) {
        return radius(d.values['attendance']);
   })
```

```javascript
// legend
var legend = svg.append("g")
	.attr("class", "legend")
	.attr("transform","translate(" + (width-100) + "," + 20 +")")
	.selectAll("g")
	.data(["Home Team", "Other"])
	.enter().append("g");
```



## Narrative Strucatures

![D4E7C067-58F2-4AB3-9E24-F34912C73855](D3js_pic/D4E7C067-58F2-4AB3-9E24-F34912C73855.png)

![6B87EC76-1B15-4304-A579-C32C80DFE1A6](D3js_pic/6B87EC76-1B15-4304-A579-C32C80DFE1A6.png)

![F940B4B4-03C5-4560-AD51-DF2B67E0BBEE](D3js_pic/F940B4B4-03C5-4560-AD51-DF2B67E0BBEE.png)

![8B7E4D98-4268-43C9-A5E9-DD011AF81D90](D3js_pic/8B7E4D98-4268-43C9-A5E9-DD011AF81D90.png)



## Make a Map

![48C7894E-E21D-49CF-91A5-4FFC4B41FA47](D3js_pic/48C7894E-E21D-49CF-91A5-4FFC4B41FA47.png)

##### Projection

利用`.mercator()`投影，三维降到二维

*  preserve equator
* stretch/ sacrifice area near poles

![C3DB191F-A1D1-40D4-B46E-B102A6E2ABE8](D3js_pic/C3DB191F-A1D1-40D4-B46E-B102A6E2ABE8.png)

```javascript
//projection, analogous to the scales, convert from logitudes and latitudes into the pixel domain
var projection = d3.geo.mercator();

//construct the SVG objects to render thoes pixels. If a projection is specified, sets the current projection to the specified projection.
var path = d3.geo.path().projection(projection);

var map = svg.selectAll('path')
              .data(geo_data.features)
              .enter()
              .append('path')
              .attr('d', path) // The path variable is actually a function that gets passed the data that is bound to each element in the selection
```

其他：![9B002BFE-45A9-4363-83F1-DF1488EFE160](D3js_pic/9B002BFE-45A9-4363-83F1-DF1488EFE160.png)



##### Geojson

"coordinates": [经度，维度]

![BAF7B176-5F16-4A6E-ABFB-E1DB56B84633](D3js_pic/BAF7B176-5F16-4A6E-ABFB-E1DB56B84633.png)



**circle area**

![BD226C84-B2DE-4257-8ACE-10219D3D2958](D3js_pic/BD226C84-B2DE-4257-8ACE-10219D3D2958.png)

```javascript

var nested = d3.nest()
               .key(function(d) {
                  return d['date'].getUTCFullYear();})
               .rollup(agg_year)
               .entries(data);
svg.append('g')
   .attr("class", "bubble")
   .selectAll("circle")
   .data(nested.sort(function(a, b) { 
      return b.values['attendance'] - a.values['attendance']; }), key_func)
   .enter()
   .append("circle")
   .attr('cx', function(d) { return d.values['x']; })
   .attr('cy', function(d) { return d.values['y']; })
   .attr('r', function(d) {
        return radius(d.values['attendance']);
   });
```

![C544A82B-66D2-4DF5-AFF8-FE06ADF12E3D](D3js_pic/C544A82B-66D2-4DF5-AFF8-FE06ADF12E3D.png)



##### 交互

![5FDFB02A-CAD7-4F82-BCB7-0692B065182D](D3js_pic/5FDFB02A-CAD7-4F82-BCB7-0692B065182D.png)



```javascript
// 一次出现一个 year对应的图
function update(year) {
          var filtered = nested.filter(function(d) {
              return new Date(d['key']).getUTCFullYear() === year;
          });

          d3.select("h2")
            .text("World Cup " + year);

          var circles = svg.selectAll('circle')
                           .data(filtered, key_func);

          circles.exit().remove();

          circles.enter()
                 .append("circle")
                 .transition()
                 .duration(500)
                 .attr('cx', function(d) { return d.values['x']; })
                 .attr('cy', function(d) { return d.values['y']; })
                 .attr('r', function(d) {
                    return radius(d.values['attendance']);
                 });

          var countries = filtered[0].values['teams'];

          function update_countries(d) {
              if(countries.indexOf(d.properties.name) !== -1) {
                  return "lightBlue";
              } else {
                  return 'white';
              }
          }

          svg.selectAll('path')
             .transition()
             .duration(500)
             .style('fill', update_countries)
             .style('stroke', update_countries);

      }
```
```javascript
// 按year设置buttons
var years = [];
        for(var i = 1930; i < 2015; i += 4) {
          if(i !== 1942 && i !== 1946) {
            years.push(i);
          };
        }
        var buttons = d3.select("body")
                        .append("div")
                        .attr("class", "years_buttons")
                        .selectAll("div")
                        .data(years)
                        .enter()
                        .append("div")
                        .text(function(d){
                            return d;
                        });
```

```javascript
// 先依次把每年的图 展示一遍（隔1000ms）；然后根据点击button的year展示相应年份的数据（if条件成立）。
var year_interval = setInterval(function() {
            update(years[year_idx]);

            year_idx++;

            if(year_idx >= years.length) {
                clearInterval(year_interval);

                var buttons = d3.select("body")
                        .append("div")
                        .attr("class", "years_buttons")
                        .selectAll("div")
                        .data(years)
                        .enter()
                        .append("div")
                        .text(function(d) {
                            return d;
                        });

                buttons.on("click", function(d) {
                    d3.select(".years_buttons")
                      .selectAll("div")
                      .transition()
                      .duration(500)
                      .style("color", "black")
                      .style("background", "rgb(251, 201, 127)");

                    d3.select(this)
                      .transition()
                      .duration(500)
                      .style("background", "lightBlue")
                      .style("color", "white");
                    update(d);
                });
            }
          }, 1000);
```

##### Martini Narrative Strucature

![5A3F3DCA-88D4-4D92-B852-F4E03C4A802A](D3js_pic/5A3F3DCA-88D4-4D92-B852-F4E03C4A802A.png)



## PS:

##### start local server

 `python -m SimpleHTTPServer 8000`

##### http protocol

http protocol —host — port—file

way to find address— apartment—unit—file

![B6FE05A2-13FA-412B-90E7-13857F2A96B6](D3js_pic/B6FE05A2-13FA-412B-90E7-13857F2A96B6.png)

##### Strict equality

![729056DB-8798-44BF-8175-159CC2048B22](D3js_pic/729056DB-8798-44BF-8175-159CC2048B22.png)

##### SVG 

**Coordinate Space**

![CA7964B8-D72B-44AA-BB1C-FFA729D66D05](D3js_pic/CA7964B8-D72B-44AA-BB1C-FFA729D66D05.png)

**g-element**

![68D4A824-80C8-4F91-84C4-4D396FE2654B](D3js_pic/68D4A824-80C8-4F91-84C4-4D396FE2654B.png)

![0A08AE50-C794-4150-B899-9CAC9DD0F976](D3js_pic/0A08AE50-C794-4150-B899-9CAC9DD0F976.png)

http://tutorials.jenkov.com/svg/g-element.html