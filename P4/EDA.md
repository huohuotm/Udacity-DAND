# EDA

Exploratory data analysis is an opportunity to let the data surprise you. 

Think back to the goals of your investigation. What question are you trying to answer? That question might be relatively simple or one dimensional like the comparison of two groups on a single outcome variable that you care about. Even in such a case, exploratory data analysis is an opportunity to learn about surprises in the data. Features of the data that might lead to unexpected results. 

It can also be an opportunity to learn about other interesting things that are going on in your data.

 What should you do first? Well certainly you want to understand the variables that are most central to your analysis, often, this is going to take the form of producing summaries and visualizations of those individual variables.



# 1. 探索单变量

##### 直方图&Summary

- 是否均匀分布：中位数，均值，极值


- 如果严重左偏/右偏 ，考虑log，sqrt变换
- 方法： 对比（histogram facet；frequency polygon；boxplot）、adjust bin width、scale transform

![屏幕快照 2017-01-25 下午4.00.39](/Users/weidian1/Documents/Study/MD_pic/屏幕快照 2017-01-25 下午4.00.39.png)

##### 异常值

判断异常值的类型，如何处理

- bad data about a non-extreme case
- bad data about an extreme case
- good data about an exteme case

![屏幕快照 2017-01-25 下午3.46.04](/Users/weidian1/Documents/Study/MD_pic/屏幕快照 2017-01-25 下午3.46.04.png)

##### 缺失值及其原因



# 2. 探索双变量

 In this lesson, we learned how to explore the relationship between two variables. 

Our main visualization tool, was the scatter plot. But we also augmented the scatter plot, with conditional summaries, like means.

We also learned about the benefits and the limitations of using correlation. To understand the relationship between two variables and how correlation may affect your decisions over which variables to include in your final models.

As usual, another important part of this lesson was learning how to make sense of data through adjusting our visualizations. 

We learned not to necessarily trust our interpretation of initial scatterplots like with the seasonal temperature data. 

And we learned how to use jitter and transparency toreduce over plotting.





### 2.1 Scatter plot

##### data aggregation(dplyr)

```
pf.fc_by_age =
  pf %>%
  group_by(age) %>%
  summarise(fc_mean = mean(friend_count),
            fc_median = median(friend_count),
            n = n()
            )%>%
  arrange(age)
```

##### Scatter plot & Conditional summaries

###### Scatter plots

- reduce overplot：alpha，jitter
- change scale: scale_y_log10() ;  coord_trans( y = "log10")
- set axis limitation: ylim(0,4000) ;  coord_cartesian(ylim=c(0,4000))

![屏幕快照 2017-01-25 下午6.03.45](/Users/weidian1/Documents/Study/MD_pic/屏幕快照 2017-01-25 下午6.03.45.png)

![屏幕快照 2017-01-25 下午7.16.05](/Users/weidian1/Documents/Study/MD_pic/屏幕快照 2017-01-25 下午7.16.05.png)



###### Conditional summaries(Overlay with Raw Data)

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

###### Smooth Noise in Conditional Means

1. 加平滑器smooth
2. 小区间内取均值

```R
library(gridExtra)
p1 = ggplot(data=subset(pf.fc_by_age_with_months,age_with_months<71)
            ,aes(x=age_with_months,y=fc_mean))+
        geom_line()+
        geom_smooth()
p2 = ggplot(data=subset(pf.fc_by_age,age<71)
            ,aes(x=age,y=fc_mean))+
        geom_line()+
        geom_smooth()
p3 = ggplot(data=subset(pf,age<71)
            ,aes(x=round(age/5)*5,y=friend_count))+
        geom_line(stat = 'summary',fun.y='mean')
        
grid.arrange(p1,p2,p3,ncol=1)
```

![屏幕快照 2017-01-25 下午9.22.48](../Documents/Study/MD_pic/屏幕快照 2017-01-25 下午9.22.48.png)

### 2.2 Correlation			

==统计笔记，相关性==



Pearson: 衡量线性关系。前提：变量是连续且服从正态分布，无outliers,散点图的形状是线性,同方差。 

 Spearman: 秩相关。data must be at least ordinal and scores on one variable must be montonically related to the other variable.    

Kendall: 



```{r Correlation}
cor(pf$age,pf$friend_count,method = 'pearson')
cor.test(pf$age,pf$friend_count,method = 'pearson')

with(pf,cor.test(age,friend_count,method = 'pearson'))	
```


### 2.3 Make sense of data

- adjusting our visualizations

```R
#install.packages('alr3')
library(alr3)
data(Mitchell)
library(dplyr)
utils::View(Mitchell)
# 按12个月 分
ggplot(data = Mitchell,aes(x=Month,y=Temp))+
  geom_point()+
scale_x_continuous(breaks = seq(0,203,11))
```

月份和温度之间无明显规律

![屏幕快照 2017-01-26 下午1.27.28](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午1.27.28.png)

按12个月分，呈现周期性规律

![屏幕快照 2017-01-26 下午1.21.11](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午1.21.11.png)

- new perspective

```R
#每月温度
ggplot(aes(x=(Month%%12),y=Temp),data=Mitchell)+ 
  geom_point() 
```

![屏幕快照 2017-01-26 下午1.22.21](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午1.22.21.png)



# 3. 探索多变量

 We started with simple extensions to the scatter plot, and plots of conditional summaries that you worked with in lesson four, such as adding summaries for multiple groups. 

Then, we tried some techniques for examining a large number of variables at once, such as scatter-plot matrices and heat maps. >>

 We also learned how to reshape data, moving from broad data with one row per case, to aggregate data with one row per combination of variables, and we moved back and forth between long and wide formats for our data.

##### Example Case

现象1：女性拥有朋友数量的平均值>男性，

现象2：但是男性要比女性稍微年轻些

问：女性的朋友数 高在哪里—> 比较女性和男性在各个年纪的中位数

答：在大部分年龄，女性的朋友数都高于男性（除异常波动）

![屏幕快照 2017-01-26 下午2.02.47](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午2.02.47.png)

推测:  Users with a longer tenure tend to have higher friend counts

虚线：整体均值

##### ![屏幕快照 2017-01-26 下午2.55.56](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午2.55.56.png)		Bias-Variance Tradeoff

权衡方差和偏差：取区间均值 7 * round(tenure / 7)



##### Looking at samples of data set

如果数据集有多个对象并且一个对象对应多条记录，建议先采样观察，特别是有时间变化。

```R
set.seed(1234)
#select 16 different household not 16 samples
sample.ids = sample(levels(yo$id),16) 
ggplot(data=subset(yo,id %in% sample.ids) 
                   ,aes(x=time,y=price))+
  geom_point(aes(size=all.purchases),fill=I('#F79420'),shape=21)+
  geom_line() +
  facet_wrap(~id)
```

![屏幕快照 2017-01-26 下午3.19.40](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午3.19.40.png)

##### Matrix Plots

批量作图，前期粗略看双变量间关系

|                | qualitative, qualitative pairs           | quantitative, quantitative pairs | qualitative, quantitative pairs |
| -------------- | ---------------------------------------- | -------------------------------- | ------------------------------- |
| Lower triangle | grouped histograms (y as grouping factor) | scatter plots                    | grouped histograms              |
| Upper triangle | grouped histograms (x as  grouping factor) | coefficient of correlation       | box plots                       |



##### Heat Maps

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

# 4. 建模

### 4.1 Linear Model

[Ref](http://data.princeton.edu/R/linearModels.html)

##### Fitting a Model

```R 
lmfit = lm( change ~ setting + effort )
```

##### Examining a Fit

$R^2$ : account for the variance of the target variable

```R
lmfit
#-----------output--------- 
##Call:
##lm(formula = change ~ setting + effort)
## 
##Coefficients:
##(Intercept)      setting       effort  
##   -14.4511       0.2706       0.9677
```

```R
summary(lmfit) #more detail
#-----------output---------  
##Call:
##lm(formula = change ~ setting + effort)
## 
##Residuals:
##     Min       1Q   Median       3Q      Max 
##-10.3475  -3.6426   0.6384   3.2250  15.8530 
## 
##Coefficients:
##            Estimate Std. Error t value Pr(>|t|)    
##(Intercept) -14.4511     7.0938  -2.037 0.057516 .  
##setting       0.2706     0.1079   2.507 0.022629 *  
##effort        0.9677     0.2250   4.301 0.000484 ***
##---
##Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
## 
##Residual standard error: 6.389 on 17 degrees of freedom
##Multiple R-Squared: 0.7381,     Adjusted R-squared: 0.7073 
##F-statistic: 23.96 on 2 and 17 DF,  p-value: 1.132e-05 
```



方差检验 anova：

```R
anova(lmfit)
#-----------output--------- 
##Analysis of Variance Table
## 
##Response: change
##          Df  Sum Sq Mean Sq F value    Pr(>F)    
##setting    1 1201.08 1201.08  29.421 4.557e-05 ***
##effort     1  755.12  755.12  18.497 0.0004841 ***
##Residuals 17  694.01   40.82                      
##---
##Signif. codes:  0  `***'  0.001  `**'  0.01  `*'  0.05  `.'  0.1  ` '  1
```

等价

```R
par(mfrow=c(2,2))
plot(lmfit)
```

This will produce a set of four plots: 

- residuals versus fitted values: used to detect non-linearity, unequal error variances, and outliers.
- Q-Q plot of standardized residuals, 
- a scale-location plot (square roots of standardized residuals versus fitted values
- a plot of residuals versus leverage that adds bands corresponding to Cook's distances of 0.5 and 1

![屏幕快照 2017-01-26 下午7.40.49](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午7.40.49.png)

##### Extracting Results

```R
fitted(lmfit)
#-----------output--------- 
##        1         2         3         4         5         6         7         8 
##-2.004026  5.572452 25.114699 21.867637 28.600325 24.146986 17.496913 10.296380 
##        ... output edited ... 

 coef(lmfit)
#-----------output---------
##(Intercept)     setting      effort 
##-14.4510978   0.2705885   0.9677137 
```



##### Updating Model

```R
m1 <- lm(I(log(price)) ~ I(carat^(1/3)), data = diamonds)
m2 <- update(m1, ~ . + carat)
m3 <- update(m2, ~ . + cut)
m4 <- update(m3, ~ . + color)
m5 <- update(m4, ~ . + clarity)
mtable(m1, m2, m3, m4, m5,sdigits = 3)
```





### 4.2 Random Forest

##### Seperate data set into train & test

##### Select features number & tree number

```R
# divide data set into train set and test set
set.seed(345888)
sample_data <- sample(2,nrow(new_wine),replace=TRUE,prob=c(0.7,0.3))
tra_data <- new_wine[sample_data==1,]
test_data <- new_wine[sample_data==2,]

# select number of features at node
for (i in 1:(ncol(new_wine)-1)){
  test_model <- randomForest(quality~.,data=new_wine,mtry=i)
  err <- mean(test_model$err)
  print(err)
}
#-----------------output--------------- choose using 10 features
##[1] 0.3177336
##[1] 0.3135958
##[1] 0.3084576
##[1] 0.3108759
##[1] 0.3068064
##[1] 0.3065039
##[1] 0.3055613
##[1] 0.305975
##[1] 0.3068713
##[1] 0.3024339
```



```R
# select tree number,170
tran_model <- randomForest(quality~.,data=new_wine,mtry=10,ntree=300)
plot(tran_model)
```

![屏幕快照 2017-01-26 下午8.59.21](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午8.59.21.png)

##### Training model

```R
# training model
tran_model <- randomForest(quality~.,data=new_wine,mtry=10,ntree=170)
tran_model
#-----------------output---------------
 
## Call:
##  randomForest(formula = quality ~ ., data = new_wine, mtry = 10,      ntree = 170) 
##                Type of random forest: classification
##                      Number of trees: 170
## No. of variables tried at each split: 10
## 
##         OOB estimate of  error rate: 28.26%
## Confusion matrix:
##    3  4    5    6   7  8  9 class.error
## 3 40  0    0    0   0  0  0   0.0000000
## 4  0 49   74   40   0  0  0   0.6993865
## 5  0 13 1048  382  14  0  0   0.2807138
## 6  0  2  278 1771 143  4  0   0.1942675
## 7  0  2   17  321 528 12  0   0.4000000
## 8  0  0    2   46  44 83  0   0.5257143
## 9  0  0    0    0   0  0 20   0.0000000
```

##### Importance of features

```R
tran_imp <- importance(x=tran_model)
varImpPlot(tran_model)
tran_imp
#---------------------output---------------
##                      MeanDecreaseGini
## fixed.acidity                280.5509
## volatile.acidity             366.2120
## citric.acid                  281.9894
## residual.sugar               324.5029
## chlorides                    296.9875
## free.sulfur.dioxide          363.8392
## pH                           315.3327
## sulphates                    289.7706
## alcohol                      484.9548
## bound.sulfur.dioxide         347.6010

```

![屏幕快照 2017-01-26 下午9.12.47](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午9.12.47.png)

##### Prediction of test data

```R
# test data predict
table(actual=test_data$quality,predicted=predict(tran_model,newdata = test_data,type = "class"))
##       predicted
## actual   3   4   5   6   7   8   9
##      3  11   0   0   0   0   0   0
##      4   0  49   0   0   0   0   0
##      5   0   0 428   0   0   0   0
##      6   0   0   0 621   0   0   0
##      7   0   0   0   0 271   0   0
##      8   0   0   0   0   0  55   0
##      9   0   0   0   0   0   0   3
```



# 附：

##### 箱线图（异常值）：

四分位数：$Q_1,Q_2（中位数）,Q_3$

IQR：四分位数全距$IQR=Q_3 - Q_1$

异常值：距离IQR上下沿（$Q_3,Q_1$）大于1.5IQR 

![screenshot](/Users/weidian1/Documents/Study/MD_pic/screenshot.png)



----

##### Pearson系数：

![screenshoewt](../Documents/Study/MD_pic/screenshoewt.png)

----

##### Bias vs Variance

[Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)

Bias: the difference between the expected (or average) prediction and the correct value

Variance: the variability of a model prediction for a given data point

![屏幕快照 2017-01-26 下午3.02.32](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午3.02.32.png)

----

##### 方差分析

http://www.statmethods.net/stats/anova.html

https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE%E5%88%86%E6%9E%90

----

##### 回归诊断

http://www.cnblogs.com/jpld/p/4453044.html

![屏幕快照 2017-01-26 下午8.52.38](../Documents/Study/MD_pic/屏幕快照 2017-01-26 下午8.52.38.png)

为理解这些图形，我们来回顾一下oLs回归的统计假设。

- 正态性：当预测变量值固定时，因变量成正态分布，则残差值也应该是一个均值为0的正态分布。正态Q-Q图(Normal Q-Q，右上)是在正态分布对应的值下，标准化残差的概率图。若满足正态假设，那么图上的点应该落在呈45度角的直线上;若不是如此，那么就违反了正态性的假设。
- 独立性：你无法从这些图中分辨出因变量值是否相互独立，只能从收集的数据中来验证。上面的例子中，没有任何先验的理由去相信一位女性的体重会影响另外一位女性的体重。假若你发现数据是从一个家庭抽样得来的，那么可能必须要调整模型独立性的假设。
- 线性：若因变量与自变量线性相关，那么残差值与预测(拟合)值就没有任何系统关联。换句话说，除了自噪声，模型应该包含数据中所有的系统方差。在“残差图与拟合图”( Residuals vs Fitted，左上)中可以清楚的看到一个曲线关系，这暗示着你可能需要对回归模型加上一个二次项。
- 同方差性：若满足不变方差假设，那么在位置尺度图(Scale-Location Graph，左下)中，水平线周围的点应该随机分布。该图似乎满足此假设。最后一幅“残差与杠杆图”(Residuals vs Leverage，右下)提供了你可能关注的单个观测点的信息。从图形可以鉴别出离群点、高杠杆值点和强影响点。



- Leverage:  An observation with an extreme value on a predictor variable is a point with high leverage. Leverage is a measure of how far an independent variable deviates from its mean. High leverage points can have a great amount of effect on the estimate of regression coefficients.
- Influence:  An observation is said to be influential if removing the observation substantially changes the estimate of the regression coefficients. Influence can be thought of as the product of leverage and outlierness. 
- Cook's distance (or Cook's D):  A measure that combines the information of leverage and residual of the observation.

