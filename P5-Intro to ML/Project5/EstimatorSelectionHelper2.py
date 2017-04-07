import pandas as pd
from numpy import mean,std
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)



class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        model_tree = SelectFromModel(ExtraTreesClassifier(random_state=32), prefit=False)
        
        preparation = [#('scaler',min_max_scaler),
                      ('selector',model_tree)]
        self.params_selector ={'selector__threshold':['0.5*mean','0.8*mean','mean','1.2*mean','1.4*mean'] }
        
        self.preparation = preparation
        self.scores_df = None

    """
    this fit fucntion also tune threshold of feature selectin(tree model)
    """
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            #params = self.params[key]
            if isinstance(self.params[key],list):
                params = [dict(param.items() + self.params_selector.items()) for param in self.params[key]]           
            elif isinstance(self.params[key],dict):
                params = dict(self.params[key].items() + self.params_selector.items())
            preparation = self.preparation
            pipe= Pipeline(preparation +[(key,model)])        
            gs = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs,verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': mean(scores),
                 'std_score': std(scores),
            }
            return pd.Series(dict(params.items() + d.items()))

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values(by=[sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        self.scores_df = df[columns]
        return df[columns]
    
    def best_estimator_model(self):
        if  isinstance(self.scores_df,pd.DataFrame):
            model_name = self.scores_df.iloc[0,0]
            best_estimator_ = self.grid_searches[model_name].best_estimator_
            best_model = self.grid_searches[model_name].best_estimator_.steps[-1][-1]
            return best_estimator_, best_model       
        else:
            raise ValueError('scores_df is None; run fit() and score_summary() first')

#-------------------------------------example-------------------------------------------
models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(random_state=32),
    'GradientBoostingClassifier': GradientBoostingClassifier()
    'SVC': SVC()
}

params1 = { 
    'ExtraTreesClassifier': { 'ExtraTreesClassifier__n_estimators': [16, 32, 50, 100] },
    'RandomForestClassifier': { 'RandomForestClassifier__n_estimators': [16, 32, 50, 100] },
    'AdaBoostClassifier':  { 'AdaBoostClassifier__n_estimators': [25, 50, 100]
                             ,'AdaBoostClassifier__learning_rate':[0.5, 1.0,1.5,2]
                             ,'AdaBoostClassifier__base_estimator':[DecisionTreeClassifier(min_samples_split=2)\
                                                                    ,DecisionTreeClassifier(min_samples_split=3)\
                                                                    ,DecisionTreeClassifier(min_samples_split=4)]\
                             ,'AdaBoostClassifier__algorithm':['SAMME', 'SAMME.R']
                           }
    'GradientBoostingClassifier': { 'GradientBoostingClassifier__n_estimators': [16, 32, 50, 100]
                                   , 'GradientBoostingClassifier__learning_rate': [0.5, 0.8, 1.0] }
    'SVC': [
        {'SVC__kernel': ['linear'], 'SVC__C': [1, 10]},
        {'SVC__kernel': ['rbf'], 'SVC__C': [1, 10], 'SVC__gamma': [0.001, 0.0001]},
    ]
}

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler

def scorer_r_p(estimator, features_test, labels_test):
    labels_pred = estimator.predict(features_test)
    pre= precision_score(labels_test, labels_pred, average='micro')
    rec = recall_score(labels_test, labels_pred, average='micro')
    if pre>0.3 and rec>0.3:
        return f1_score(labels_test, labels_pred, average='macro')
    elif  pre>0.3 and rec<0.3:
        return 0.3
    elif rec >0.3 and pre<0.3:
        return 0.3
    return 0


from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, n_iter=50, test_size = 0.2, random_state=42)

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(features, labels,cv=sss, scoring=scorer_r_p, n_jobs=-1)
helper1.score_summary(sort_by='min_score')
best_estimator_, best_model = helper1.best_estimator_model()
