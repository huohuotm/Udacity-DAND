## Data Story

泰坦尼克船上的891个人中，只有342个人幸存，整体生存率仅仅为38% 。但另一方面，不同人群的生存率却差异很大。女人比男人和小孩，有更高的存活概率。一等舱乘员的生存率也明显高于二等舱和三等舱。

ps：数据来源kaggle数据集。我对数据进行过处理，定义小孩为年龄小于10岁的人。



## Design

我打算画3个柱状图。因为柱状图很好理解，并且容易对比各柱子代表的数值。

第一个柱状图展示整体船员生存和遇难的情况。第二个和第三个堆积柱状图用来表达不同人群生存概率的区别。

用X轴标签区分不同人群；用Y轴表示人数。在第二个和第三个图中，用柱子填充的颜色来区别是否存活；



## Feedback and Iteration

1-3反馈来自室友和同学; 4反馈，来自code reviewer。

* 反馈1：缺乏背景说明，不知道数据讲的是怎么样的“故事”

  改进：采纳了该意见。在大标题下方增加了事件背景说明，以及数据来源。

* 反馈2：一下子看到三幅图，有点困惑。三个图之间有什么关系吗？

  改进：三个图一口气的展示方式确实会让人懵圈。我改为“总-分”的展示方式。在初始页面，读者第一眼只会看到一幅图（整体存活情况）。点击“more”按钮之后，第一幅图缩小，并且出现另外两个图。循序渐进的方式能让读者更好地探索不同人群、不同船舱等级的生存率差异。

* 反馈3：在后两个图中没有图例，必须把鼠标移到图上看了悬浮窗的内容才知道这块数据是代表了哪个群体；

  改进：移动鼠标确实可以获取群体信息，但是用户体验不好，不能直观的得到群体信息。增加图例不会占用大量空间，确可以大大增加可读性，所以我会添加图例。

*  反馈4：在可视化展示部分，建议增加对children的定义。

   改进：在第一次提交的项目中，并没有在可视化作品中说明children的定义。在第二次提交的项目中，修改了图二的X轴坐标`children` —> `children (age<10) ` ，但是reviewer说没有看到修改。难道是不明显？

*  反馈5：认为将乘客分为“女人，男人，儿童”这点不是很恰当，建议按照性别进行分组来分析。

   不采纳原因：性别的确也是一个不错的分析角度。但是，这里我分析的角度是人群不是性别，基于在危机时刻有先救妇女和孩子的文化背景考虑的。

* 反馈6：在background中把你分析的思路简要说明一下，这样可以使读者了解你的分析思路，可视化也不会引起读者的困惑。

  改进：挺有道理，不清楚分析思路的情况下，容易产生困惑。我在background中说明了数据探索的角度，也对三个图进行了简单的说明。

  ​



## Reference

https://www.kaggle.com/c/titanic

http://d3js.org/

http://dimplejs.org/

https://github.com/pmsi-alignalytics/dimple/

https://en.wikipedia.org/wiki/RMS_Titanic

https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API

http://www.w3school.com.cn/svg/svg_rect.asp

http://alignedleft.com/tutorials/d3/adding-elements

http://www.ourd3js.com/wordpress/883/