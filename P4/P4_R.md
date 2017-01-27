

# Intro to R

## Vector

```R
udacious <- c("Chris Saden", "Lauren Castellano",        
		 "Sarah Spikes","Dean Eckles",
          "Andy Brown", "Moira Burke",
          "Kunal Chawla")

numbers <- c(1:10)
numbers
numbers <- c(numbers, 11:20)
numbers

mystery = nchar(udacious)  #a vector that contains the number of characters
mystery
mystery == 11  # boolean vector
udacious[mystery == 11] # subset our udacious vector
```


## Data

```R
data(mtcars)
names(mtcars)
?mtcars  #help
mtcars
str(mtcars) #structure of data
dim(mtcars) #dimension of data
summary(mtcars) 
?row.names
row.names(mtcars) <- c(1:32)

data(mtcars)
head(mtcars, 10)
tail(mtcars, 3)
mtcars$mpg #$ select columns
mean(mtcars$mpg)
```



## Regular operator

##### read file

##### select; subset

##### ifelse

##### logical operator

##### remove

##### install package

##### cut continuous variable

```{r}
# work path
getwd()
setwd("path")
read.csv("filename",sep=",")

# select 
efficient = subset(mtcars,mpg >=23)
dim(efficient)
str(efficient)
subset(mtcars, mpg > 30 & hp > 100) #logical operator
mtcars <- subset(mtcars,select =  -year) #- drop

# ifelse
mtcars$wt
cond <- mtcars$wt < 3
cond
mtcars$weight_class <- ifelse(cond, 'light', 'average')
mtcars$weight_class
cond <- mtcars$wt > 3.5
mtcars$weight_class <- ifelse(cond, 'heavy', mtcars$weight_class)
mtcars$weight_class
rm(cond)

# install package
install.packages('knitr', dependencies = T)
library(knitr)

# cut quantitate variable
pf$year_joined.bucket = 
    cut(pf$year_joined,breaks = c(2004,2009,2011,2012,2014))
```




# Intro to RMarkDown

语法同MarkDown，需要`library(knitr)`。

## New chunk

insert chunk (command+option+i)，write code，run(command+shift)。

```R
​```{r}
code
​```
```

## Setting 

##### set chunk options

```R
​```{r chunk_name,echo=TRUE,message=FALSE,warning=FALSE,fig.height=4,fig.width=5}
code
​```
```

##### set global options

```R
​```{r set_global}
knitr::opts_chunk$set(echo=TRUE)
​```
```

![chunk_option](../Documents/Study/MD_pic/chunk_option.png)





# Useful Packages & Methods

## Intro to "Dplyr"

[dplyr&tidyr](https://dl.dropboxusercontent.com/u/5896466/wrangling-webinar.pdf)

View(Capital V)

pipe operator %>%

`tb %>% select(child:elderly)`等价`select(tb,child:elderly)`

##### Access Data

| Function  |                                          |
| --------- | ---------------------------------------- |
| select()  | `select(storms, storm, pressure)` `select(storms,-storm)` `select(storms,wind:date)` |
| filter()  | `filter(storms,condtion)` 可以用logical operator |
| mutate()  | `mutate(storms, ratio = pressure/wind, inverse = ratio^-1)`  添加新变量并直接可用 |
| summarise | `pollution %>% summarise(mean = mean(amount), sum = sum(amount), n = n())` 聚合计算 |
| arrange   | `arrange(storms,wind,data)` 默认asc; `arrange(storms,desc(wind))` |
| group_by  | ungroup()                                |

```R
tb %>%
  group_by(country, year) %>%
  summarise(cases = sum(cases)) %>%
  summarise(cases = sum(cases))
```

![屏幕快照 2017-01-27 下午1.31.36](../Documents/Study/MD_pic/屏幕快照 2017-01-27 下午1.31.36.png)



##### Join Data	

| Function        |       |
| --------------- | ----- |
| bind_cols()     | 列相加   |
| bind_rows()     | 行相加   |
| dplyr::unions() | 行去重   |
| intersect()     | 取交集   |
| setdiff()       | 交集的补集 |
| left_join()     |       |
| inner_join()    |       |

​	

## Intro to "Tidyr"

http://www.rstudio.com/resources/cheatsheets/

A package that reshapes the layout of data sets

| function   |                                          |
| ---------- | ---------------------------------------- |
| gather()   | wide -> long                             |
| spread()   | long -> wide                             |
| unite()    | `unite(storms2, "date", year, month, day, sep = "-")` |
| separate() | `separate(storms, date, c("year", "month", "day"), sep = "-")` |

![屏幕快照 2017-01-27 下午1.16.20](../Documents/Study/MD_pic/屏幕快照 2017-01-27 下午1.16.20.png)

![屏幕快照 2017-01-27 下午1.13.48](../Documents/Study/MD_pic/屏幕快照 2017-01-27 下午1.13.48.png)



##### Reshape Package

| Function |               |
| -------- | ------------- |
| melt()   | wide -> loong |
| dcast()  | long -> wide  |





## Intro to Date-Format

[date-format](https://www.r-bloggers.com/date-formats-in-r/)

Note:

If your data were exported from Excel, they will possibly be in numeric format. Otherwise, they will most likely be stored in character format.



##### Importing Dates

If your data were exported from Excel, they will possibly be in numeric format. Otherwise, they will most likely be stored in character format.



Importing Dates from Character Format

```R
dates <- c("05/27/84", "07/07/05")
betterDates <- as.Date(dates, format = "%m/%d/%y")  #provide dates formats

betterDates
#---------------output--------------
#[1] "1984-05-27" "2005-07-07"
```



##### Importing Dates from Numeric Format

```R
# from Windows Excel:
dates <- c(30829, 38540)
betterDates <- as.Date(dates, origin = "1899-12-30")  #the origin date that Excel starts counting from

betterDates
#---------------output--------------
#[1] "1984-05-27" "2005-07-07"

# from Mac Excel:
dates <- c(29367, 37078)
betterDates <- as.Date(dates,origin = "1904-01-01")

betterDates
#---------------output--------------
#[1] "1984-05-27" "2005-07-07"
```



##### Changing Date Formats

```R
format(betterDates, "%a %b %d")
#---------------output--------------
#[1] "Sun May 27" "Thu Jul 07"
```

[other-format](https://www.r-bloggers.com/date-formats-in-r/)

**Lubridate is an R package that makes it easier to work with dates and times**



# Refs

http://www.cookbook-r.com/

https://www.rstudio.com/wp-content/uploads/2015/02/rmarkdown-cheatsheet.pdf

[dplyr&tidyr](https://dl.dropboxusercontent.com/u/5896466/wrangling-webinar.pdf)

