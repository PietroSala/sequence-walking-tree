from sequence_walking_tree import *
from tqdm import tqdm
import random 


class SequenceWalkingForest():

    def __init__(self, number_of_trees = 10, tree_parameters = {}):
        self.number_of_trees = number_of_trees
        self.tree_parameters = tree_parameters
        self.trees        = []
        self.tree_weights = []
        self.tree_confusion_matrices = []
        self.tree_ranks = []

    @staticmethod
    def compute_tree_rank(t): 
        t_diam, t_ray = SequenceWalkingForest.compute_tree_rank(t[True]) if True in t else (0, 0)
        f_diam, f_ray = SequenceWalkingForest.compute_tree_rank(t[False]) if False in t else (0, 0)
        return  max(t_diam, f_diam, t_ray + f_ray + 1), max(f_ray, t_ray) + 1
    
    @staticmethod
    def versus_order(s_1,s_2):
        r  = []
        for i in s_1:
            for j in s_2:
                if i < j or (i != j and j not in s_1 ):
                    r.append((i,j))
        r.sort(key=lambda x: (x[0], x[1]))
        return r

    def fit(self, z, weight_index = 0):
        z_size = len(z.raw_x)
        id_set = {i for i in range(z_size)}.difference({ i for i,_ in  z.excluded_positions(weight_index) })
        id_list = list(id_set)
        train_size = len(id_set)
        for i in range(self.number_of_trees):
            train_index = - (2*i + 1)
            score_index = - (2*i + 2)
            z.sample_df[train_index] = 0
            z.sample_df[score_index] = 0
        for i in tqdm(range(self.number_of_trees)):
            train_index = - (2*i + 1)
            score_index = - (2*i + 2)
            for i in range(len(z.sample_df)):
                if i not in id_set:
                    z.sample_df.loc[i, score_index] = 1
            for _ in range(train_size):
                j = random.choice(id_list)
                z.sample_df.loc[j, train_index] = z.sample_df.loc[j, train_index] + 1
                z.sample_df.loc[j, score_index] = 1
            model = self.fit_tree(z, train_index, score_index)    
            self.trees.append(model)
            self.tree_weights.append(self.compute_weight(z,score_index))
            self.tree_confusion_matrices.append(self.compute_confusion_matrix(z,score_index))
            self.tree_ranks.append(SequenceWalkingForest.compute_tree_rank(model.tree))

    def fit_tree(self, z, train_index, score_index):
        r = SequenceWalkingTree(**self.tree_parameters)
        r.fit(z, train_index)
        z.predictions[score_index] = r.predict(z, score_index)
        self.compute_weight(z, score_index)
        return r
    
    def class_prediction_merge(self, z, score_index):
        return pd.merge(z.sample_df[['class']], z.predictions[score_index][['prediction']], left_index=True, right_index=True, how='inner')

    def compute_weight(self, z, score_index):
        merged = self.class_prediction_merge(z, score_index)
        merge_series = merged["class"] == merged["prediction"]
        if merge_series.sum() == 0:
            return 0
        else:
            return merge_series.mean()    
    
    def compute_confusion_matrix(self, z, score_index):
        c = list(z.classes)
        c.sort()
        merged = self.class_prediction_merge(z, score_index)
        r = pd.DataFrame([{ "prediction":  x, "class": y, "count": 0  } for x in c for y in c  ])
        r.set_index(["prediction", "class"], inplace=True,drop=True)
        for i in range(len(merged)):
            x, y = merged.iloc[i]["prediction"], merged.iloc[i]["class"]
            r.loc[(x,y), "count"] += 1
        return r
    
    def predict(self,z, weight_index = 0, start_index=1, include_class = True, keep_predictions = False):
        all_predictions = [] 
        for i in range(len(self.trees)):
            z.sample_df[start_index + i] = z.sample_df[weight_index]
            z.predictions[start_index + i] = self.trees[i].predict(z, start_index + i)
            if i == 0:
                all_predictions = z.predictions[start_index + i][["prediction"]]
            else:
                all_predictions = pd.merge(all_predictions, z.predictions[start_index + i][["prediction"]], suffixes=(f"_{i-1}", f"_{i}"), left_index=True, right_index=True, how="inner")
         
        voting = pd.merge(SequenceWalkingForest.majority(all_predictions), self.weighted(all_predictions), left_index=True, right_index=True, how='inner')
        voting = pd.merge(voting, self.track_record(all_predictions), left_index=True, right_index=True, how='inner')
        if keep_predictions:
           voting = pd.merge(voting, all_predictions, left_index=True, right_index=True, how='inner' )
        if include_class:
            excluded_samples ={i for i, _ in z.excluded_positions(weight_index)}
            excluded_list = [j in excluded_samples  for j in range(len(z.sample_df)) ]
            voting = pd.merge(z.sample_df[excluded_list]["class"],voting, left_index=True, right_index=True, how='inner')
        z.forest_prediction[weight_index] = voting
        return voting   

    @staticmethod
    def majority(all_predictions):   
        r = all_predictions.apply(lambda row: pd.Series(list(row)).value_counts().idxmax(), axis=1) 
        r.name = "majority"
        return r   

    def best_weight(self, list):
        w = {c:0 for c in set(list)}
        for i, c in enumerate(list):
            w[c] = w[c] + self.tree_weights[i]
        return max(w,key=lambda k: w[k])

    def weighted(self, all_predictions):
        r = all_predictions.apply(lambda row: self.best_weight(list(row)), axis=1)
        r.name = "weighted"
        return r
    
    def best_track_record(self,list):
        w = {c:0 for c in set(list)}
        for i, c in enumerate(list):
            col = self.tree_confusion_matrices[i].loc[c]["count"]
            if col.sum() > 0:
                probability = col/col.sum()
                for k in w.keys():
                    w[k] = w[k] + probability.loc[k]
        return max(w,key=lambda k: w[k])    
            
    def track_record(self, all_predictions):
        r = all_predictions.apply(lambda row: self.best_track_record(list(row)), axis=1)
        r.name = "track_record"
        return r
    
    def similarity(z, weight_index = 0, weight_index_versus = 0):
        pass