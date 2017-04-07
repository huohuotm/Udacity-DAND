**P5-Identify Fraud.zip** is the file that I submited to review, including three pickle files (<u>my_dataset.pkl</u> , <u>my_classifier.pkl</u> , <u>my_feature_list.pkl</u>) , a python file <u>poi.py</u> generating these three pickle files,  a pdf file  <u>Document of work.pdf</u> describing analysis process  and a  text file <u>reference.txt</u> listing references.

**final_project** file contains relative code and data 

<u>Eron_poi_id.ipynb</u> is equal to <u>poi_id.py</u>, but in python notebook format.

<u>final_project_dataset.pkl</u> is origin dataset.

<u>**EstimatorSelectionHelper.py**</u> create a helper class for running paramater grid search across different classification or regression models. I modified \__init\__ and  _fit_ fucntion in <u>EstimatorSelectionHelper2.py</u> , and the helper class can also tune the threshold of feature selectin(tree model) . A lot of parameters and models make computing very complicated and slow. I strongly suggest that use this method with a clear purpose.

**tools** file contains some python models used in final project.

[start_code](https://github.com/udacity/ud120-projects)

[EstimatorSelectorHelper-ref](http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/) 

