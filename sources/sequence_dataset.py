from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
from random import sample

class DataFrameHelper():

    @staticmethod
    def df_column_to_dict(df,  column):
       return  {k:v[column] for k, v in  df.to_dict(orient="index").items()}

class LossHelper():

    @staticmethod
    def information_gain(weighted_df_1, weighted_df_2):
        weighted_df = pd.concat([weighted_df_1, weighted_df_2], axis=1)
        weighted_df.fillna(0, inplace=True)
        weighted_df = weighted_df.T.groupby(weighted_df.columns).sum().T
        w, en = LossHelper.weighted_entropy(np.array(weighted_df.weight))
        w1, en1 =  LossHelper.weighted_entropy(np.array(weighted_df_1.weight))
        w2, en2 =  LossHelper.weighted_entropy(np.array(weighted_df_2.weight))
        return en - ((w1/w)*en1 + (w2/w)*en2)
    
    @staticmethod
    def weighted_entropy(weights):
        total = np.sum(weights)
        weights = weights / total
        entropy_val = -np.sum(weights * np.log2(weights, where=(weights > 0)))
        return total, entropy_val


class SequenceDataset():
    
    def __init__(self):
        pass

    def fit(self, x, y, w = None):
        self.raw_x, self.raw_y = x, y 
        self.raw_w = w if w is not None else [1 for _ in  range(len(x))]
        self.classes = set(y)

        index = pd.MultiIndex.from_arrays([[], [], []], names=['sample', 'position','label'])
        next_df = pd.DataFrame(index=index, columns= ['next_position', 'distance'])
        for i in range(len(self.raw_x)):
            seq = self.raw_x[i]
            for j1 in range(len(seq)):
                if j1 > 0:
                    for j2 in range(j1):
                        index = (i, j2, seq[j1][1])
                        if index not in next_df.index:
                            data = {'next_position': j1, 'distance': seq[j1][0] - seq[j2][0]}
                            next_df.loc[index] = data
        self.next_df = next_df

        index = pd.Index([], name='sample')
        sample_df = pd.DataFrame(index=index, columns=["class", 0])
        for i in range(len(self.raw_x)):                    
            data = {"class": self.raw_y[i], 0: self.raw_w[i]}
            sample_df.loc[i] = data
        self.sample_df = sample_df    

        index = pd.MultiIndex.from_arrays([[], []], names=['sample', 'position'])
        value_df = pd.DataFrame(index=index, columns= ['value'])
        for i in range(len(self.raw_x)):
            seq = self.raw_x[i]
            for j in range(len(seq)):
                value_df.loc[(i,j),:] = {'value': seq[j][2]}
        self.value_df = value_df  

        index = pd.Index([], name='sample')
        last_position_df = pd.DataFrame(index=index, columns=["last_position"])
        for i in range(len(self.raw_x)):                    
            data = {"last_position": len(self.raw_x[i]) - 1}
            last_position_df.loc[i] = data
        self.last_position_df = last_position_df


        self.next_df.sort_index(inplace=True)
        self.sample_df.sort_index(inplace=True)
        self.value_df.sort_index(inplace=True)
        self.last_position_df.sort_index(inplace=True)

        self.predictions = {}

    def add_prediction_dataframe(self, weight_index):
        index = pd.Index([], name='sample')
        prediction_df = pd.DataFrame(index=index, columns=["prediction", "path", "nodes", "positions" ,"end_of_sequence"])
        for i in range(len(self.raw_x)):                    
            data = {"prediction": None, "path": [], "nodes": [], "positions": [], "end_of_sequence": None}
            prediction_df.loc[i] = data
        prediction_df.sort_index(inplace=True)
        self.predictions[weight_index] = prediction_df    

    def select_next(self, sample_position_pairs):
        filtered_position_pairs = self.filter_out_last_positions(sample_position_pairs)
        if len(filtered_position_pairs) == 0:
            dummy_index = pd.MultiIndex.from_arrays([[], [], []], names=['sample', 'position','label'])
            dummy_next_df = pd.DataFrame(index=dummy_index, columns= ['next_position', 'distance'])
            return dummy_next_df
        return pd.concat([ self.next_df.xs((s,p), drop_level=False) for s,p in filtered_position_pairs], axis=0, verify_integrity=True)
    
    def select_value(self, sample_position_pairs):
        return pd.concat([ self.value_df.loc[[(s,p)]] for s,p in sample_position_pairs], axis=0, verify_integrity=True)
    
    def compute_probabilities(self, weighted_df):
        r_df  =pd.DataFrame([{"class":c, "probability":0.0} for c in self.classes])
        r_df.set_index('class', inplace=True)
        total = float(weighted_df.weight.sum())   
        for c in weighted_df.index:
            r_df.loc[c].probability = weighted_df.loc[c].weight/total
        return r_df
    
    def get_weights(self, ids, weight_index=0, ignore_class= False):
        if ignore_class:
            r = self.sample_df[self.sample_df.index.isin(ids)][weight_index].sum()
        else:    
            r = self.sample_df[self.sample_df.index.isin(ids)].groupby(["class"]).sum(weight_index)
            r.rename({weight_index:"weight"}, inplace=True, axis=1)
        return r
    
    def split_by_test(self, test_type, test, current_positions, weight_index=0, loss_function=LossHelper.information_gain):
        if test_type == "event":
            true_samples, removed, added = SequenceDataset.true_event_test(self.select_next(current_positions), *test)
        elif test_type == "value":
            true_samples, added = SequenceDataset.true_value_test(self.select_value(current_positions), test)
            removed = added
        false_samples = set([x[0] for x in current_positions]).difference(true_samples)
        r_loss = loss_function(self.get_weights(true_samples, weight_index), self.get_weights(false_samples, weight_index)) 
        return r_loss, added, current_positions.difference(removed)    
    
    def max_by_test(self, test_type, current_positions,  weight_index=0, loss_function=LossHelper.information_gain, random_samples = None):
        if test_type == "event":
            test_set = SequenceDataset.event_tests(self.select_next(current_positions), random_samples)
        elif test_type == "value":
            test_set = SequenceDataset.value_tests(self.select_value(current_positions), random_samples)
        best_value, best_test, best_true, best_false = 0, None, None, None
        for test in test_set:
            current_value, current_true, current_false = self.split_by_test(test_type,test, current_positions, weight_index, loss_function)
            if current_value > best_value:
                best_value, best_test, best_true, best_false = current_value, test, current_true, current_false
        best_true_weights  = self.get_weights([i for i,_ in best_true], weight_index,ignore_class=True) if best_true is not None else 0
        best_false_weights = self.get_weights([i for i,_ in best_false], weight_index,ignore_class=True) if best_false is not None else 0
        return best_value, best_test, best_true, best_false, best_true_weights, best_false_weights       
    
    def init_positions(self, weight_index = 0):
        return {(i,0) for i in self.sample_df[self.sample_df[weight_index] > 0].index}
    
    def excluded_positions(self, weight_index = 0):
        return {(i,0) for i in self.sample_df[self.sample_df[weight_index] == 0].index}
    
    def filter_out_last_positions(self, current_positions):
        return current_positions.difference(set(filter(lambda p: self.last_position_df.loc[p[0],"last_position"] == p[1], current_positions)))

    def filter_out_inside_positions(self, current_positions):
        return set(filter(lambda p: self.last_position_df.loc[p[0],"last_position"] == p[1], current_positions))


    @staticmethod
    def event_tests(select_next_df, random_samples = None):
        if random_samples is not None:
            l = len(select_next_df)
            sids = sample(range(l), k=min(random_samples,l))
            return set([ (r[0][2], r[1].distance) for r in select_next_df[[ i in sids for i in range(l)]].iterrows()])
        return set([ (r[0][2], r[1].distance) for r in select_next_df.iterrows()])
    
    @staticmethod
    def value_tests(select_value_df, random_samples = None):
        if random_samples is not None:
            l = len(select_value_df)
            sids = sample(range(l), k=min(random_samples,l))
            return set([ r[1].value for r in select_value_df[[ i in sids for i in range(l)]].iterrows()])
        return set([ r[1].value for r in select_value_df.iterrows()])
    
    @staticmethod
    def true_event_test(select_next_df, event, distance):
        label_selected = select_next_df.loc[(slice(None), slice(None), event)]
        distance_selected = label_selected[label_selected.distance <= distance]
        samples, removed, added = set(), set(), set()
        for r in distance_selected.iterrows():
            samples.add(r[0][0])
            removed.add((r[0][0], r[0][1]))
            added.add((r[0][0], r[1].next_position)) 
        return samples, removed, added

    @staticmethod
    def true_value_test(select_value_df, value):
        value_selected = select_value_df[select_value_df.value <= value]
        samples,  added = set(),set()
        for r in value_selected.iterrows():
            samples.add(r[0][0])
            added.add((r[0][0], r[0][1])) 
        return samples, added