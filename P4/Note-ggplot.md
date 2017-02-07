# ggplot

> "ggplot2 is designed to work in a layered fashion, starting with a layer showing the raw data then adding layers of annotationand statistical summaries. "
>
> H.Wickham



## 基本画法：

**数据底层+几何对象geom+统计变换stats+坐标系coord+分面facet**

Example:

```R
ggplot(data = diamonds,aes(x= cut,y= price))+  
     geom_point()+ 														  # 几何对象 
     stat_summary(fun.y = "median", colour = "red", size = 2, geom = "point")+ # 统计变换
     coord_cartesian(ylim=c(0,quantile(diamonds$price, probs = .99)))+ 		  # 坐标系
     facet_wrap(~color, ncol=2)											   # 分面
```

- 几何对象geom 点，线，条形
  - geom_point/geom_polygon/geom_tile/geom_jitter/geom_hline/geom_bar/geom_boxplot
- 图形属性aes ：颜色，形状，大小，透明度
- 统计变换stats：对数据进行的某种汇总
  - mean/median/max/min/ 自定义函数
- 标度scale：将数据的取值映射到图形空间，如颜色，大小，形状表示不同的取值；展示：图例【不太理解】
- 坐标系coord：设置坐标轴显示范围/数值转换/坐标轴类型（直角坐标系，极坐标轴）
  - ylim(0,100) ; coord_cartesian(ylim = c(0,100))【直接切图，不重新计算】
  - x = log(price) ；coord_trans(y= "log10") 【可自定义函数】
- 分面facet ：绘图窗口划分为若干个子窗口



## 用途

### 增加维度：

- color

  直接填充`geom_point(aes(fill='black',color=cut))`

- facet

  `  facet_wrap(~class, scales = "free", nrow = 2)`

  `facet_grid(am ~ gear, switch = "both")`

- grid

  `grid.arrange(p1,p2,p3,p4,ncol=2)`

- conditional summary（对比）

  ```R
  q = ggplot(aes(x = age, y = friend_count), data = pf)+ 
    geom_point(alpha=1/20,position = position_jitter(h=0),color='orange')+ 
    xlim(13,113)+
    coord_trans(y = "sqrt")
  q = q + geom_line(stat = 'summary',fun.y='mean')
  q = q + geom_line(stat = 'summary', fun.y = quantile,fun.args = list(probs = .9),color='blue',linetype=2)
  q = q + geom_line(stat = 'summary', fun.y = quantile,fun.args = list(probs = .5),color='blue')
  q = q + geom_line(stat = 'summary', fun.y = quantile,fun.args = list(probs = .1),color='blue',linetype=2)
  q
  ```

  ![屏幕快照 2017-01-25 下午8.41.25](../Documents/Study/MD_pic/屏幕快照 2017-01-25 下午8.41.25.png)

### 美化润色：

##### Binwidth & Axis

组间距`geom_histogram(binwidth=10)`

Axis:

```R
scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),breaks = c(350, 1000, 5000, 10000, 15000))

coord_trans(y= "log10")
xlim(0,100)
coord_cartesian(ylim = c(0,100))
```

##### Color

Default color/RColorBrewer/Manually defined/Continuous colors【见附】

```R
#overwrites the alpha,size of legend
scale_color_brewer(type = 'div',
    guide = guide_legend(title = 'Clarity', reverse = T,
    override.aes = list(alpha = 1, size = 2))) 
```



##### Jitter & Alpha

```R
geom_point(alpha = 0.5, size = 1, position = 'jitter')
geom_jitter(position = "jitter",height = 0)  #纵向不抖动
```



##### Scale

```R
#多种方法
coord_trans(y='sqrt')
scale_y_sqrt()
qplot(carat, price, data = d, log="xy")
scale_y_continuous(trans = log10_trans()) #函数可以自定义

#Create a new function to transform the carat variable
cuberoot_trans = function() trans_new('cuberoot', transform = function(x) x^(1/3),
                                      inverse = function(x) x^3)
#Use the cuberoot_trans function                                      
ggplot(aes(carat, price), data = diamonds) + 
  geom_point() + 
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3),
                     breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000),
                     breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat')
```



**Transforming the Scales vs Transforming the Coordinate System**

1. Scale transformation occurs **before** statistics, and coordinate transformation afterwards.
2. Coordinate transformation also changes the shape of geoms.

```R
d <- subset(diamonds, carat > 0.5)
p1 = qplot(carat, price, data = d) + 
  geom_smooth(method="lm")
p2 = qplot(carat, price, data = d, log="xy") + 
  geom_smooth(method="lm")
p4 = qplot(log10(carat), log10(price), data = d) + 
  geom_smooth(method="lm")
p3 = qplot(carat, price, data = d) +
  geom_smooth(method="lm") +
  coord_trans(x = "log10", y = "log10")
grid.arrange(p1,p2,p3,p4,ncol=2)
```

左上p1是原图；左下p3经坐标系变换；右上p2 右下p4，是scale变换

1. 改映射aes中x,y的值，坐标轴的数值，label才会变（右下）
2. 左下坐标系变换后，几何形状变化

![屏幕快照 2017-01-27 上午12.07.51](../Documents/Study/MD_pic/屏幕快照 2017-01-27 上午12.07.51.png)



##### Title & Axis label

```R
# Method1
ggtitle("Log_10 Residual Sugar of Wine")+
theme(plot.title = element_text(hjust = 0.5))+
ylab('Number of Wine')+
xlab("Residual Sugar - g/dm^3")
```

-----



## 工具

### Matrix Plot

看双变量间关系

**Method1**

```R
library(GGally)
set.seed(20022012)
diamond_samp <- diamonds[sample(1:length(diamonds$price), 10000), ][,c(1:7)]
ggpairs(diamond_samp, #axisLabels = 'internal'
        lower = list(continuous = wrap("points", shape = I('.'))), 
        upper = list(combo = wrap("box", outlier.shape = I('.'))))
```

|                | qualitative, qualitative pairs           | quantitative, quantitative pairs | qualitative, quantitative pairs |
| -------------- | ---------------------------------------- | -------------------------------- | ------------------------------- |
| Lower triangle | grouped histograms (y as grouping factor) | scatter plots                    | grouped histograms              |
| Upper triangle | grouped histograms (x as  grouping factor) | coefficient of correlation       | box plots                       |



![屏幕快照 2017-01-27 上午12.46.38](../Documents/Study/MD_pic/屏幕快照 2017-01-27 上午12.46.38.png)

**Method2**

```R
library(psych) 
# 好像只能画连续变量
pairs.panels(wine[ ,c('density','residual.sugar','alcohol','total.sulfur.dioxide'
              ,'free.sulfur.dioxide','bound.sulfur.dioxide','fixed.acidity')],pch=".")
```



![屏幕快照 2017-01-27 上午12.40.05](../Documents/Study/MD_pic/屏幕快照 2017-01-27 上午12.40.05.png)



### Heat Maps 

geom_tile呈方块状：uses the center of the tile and its size (x, y, width, height)

```{r}
nci <- read.table("nci.tsv")
colnames(nci) <- c(1:64)

nci.long.samp <- melt(as.matrix(nci[1:200,]))  #wide -> long
names(nci.long.samp) <- c("gene", "case", "value")  
head(nci.long.samp)

ggplot(aes(y = gene, x = case, fill = value), 
  data = nci.long.samp) +
  geom_tile() +
  scale_fill_gradientn(colours = colorRampPalette(c("blue", "red"))(100))
```

![屏幕快照 2017-01-26 下午3.56.37](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午3.56.37.png)



# 附

## Color

[ref-color-cookbook](http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/)

##### Default color selection

The default color selection uses `scale_fill_hue()` and `scale_colour_hue()`.  Adding those commands is redundant.

```R
# These two are equivalent; by default scale_fill_hue() is used
ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity")
# ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") + scale_fill_hue()
```



##### Palettes: Color Brewer

| parameter | usage                                    |
| --------- | ---------------------------------------- |
| type      | One of seq (sequential), div (diverging) or qual (qualitative) |
| palette   | If a string, will use that named palette. If a number, will index into the list of palettes of appropriate type |

RColorBrewer palette chart` display.brewer.all()`

![屏幕快照 2017-01-26 下午11.14.56](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午11.14.56.png)

```R
ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") +
    scale_fill_brewer()

ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") +
    scale_fill_brewer(palette="Set1")

ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") +
    scale_fill_brewer(palette="Spectral")

# Note: use scale_colour_brewer() for lines and points
```

![屏幕快照 2017-01-26 下午10.56.19](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午10.56.19.png)

##### Palettes: manually-defined

```R
ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") + 
    scale_fill_manual(values=c("red", "blue", "green"))

ggplot(df, aes(x=cond, y=yval, fill=cond)) + geom_bar(stat="identity") + 
    scale_fill_manual(values=c("#CC6666", "#9999CC", "#66CC99"))

# Note: use scale_colour_manual() for lines and points
```



![屏幕快照 2017-01-26 下午10.57.47](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午10.57.47.png)

##### Continuous colors

```R
# Generate some data
set.seed(133)
df <- data.frame(xval=rnorm(50), yval=rnorm(50))
# Make color depend on yval
ggplot(df, aes(x=xval, y=yval, colour=yval)) + geom_point()
# Use a different gradient
ggplot(df, aes(x=xval, y=yval, colour=yval)) + geom_point() + 
    scale_colour_gradientn(colours=rainbow(4))
```

![屏幕快照 2017-01-26 下午10.59.44](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午10.59.44.png)

![senshot](../Documents/Study/MD_pic/senshot.png)

![screenshdfot](../Documents/Study/MD_pic/screenshdfot.png)



## Package 函数覆盖问题

使用plyr中的count，需要先detach dplyr

```R
detach("package:dplyr", unload=TRUE)
library(plyr)
library(dplyr)
```

