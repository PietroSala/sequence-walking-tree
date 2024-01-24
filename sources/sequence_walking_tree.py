from sequence_dataset import *

class SequenceWalkingTree():

    def __init__(self, 
                 event_loss = LossHelper.information_gain, 
                 value_loss = LossHelper.information_gain, 
                 min_node_samples=0, 
                 max_height=float('inf'), 
                 max_value_streak=1, 
                 max_alternation=float('inf'), 
                 early_removal=True, 
                 early_stop= True,
                 alternate_if_possible= False,
                 random_samples = None,
                 best_fallback_on_random_fail = False):
        self.event_loss       = event_loss         
        self.value_loss       = value_loss        
        self.min_node_samples = min_node_samples 
        self.max_height       = max_height
        self.max_value_streak = max_value_streak 
        self.max_alternation  = max_alternation 
        self.early_removal    = early_removal    
        self.early_stop       = early_stop 
        self.alternate_if_possible = alternate_if_possible      
        self.random_samples = random_samples
        self.best_fallback_on_random_fail = best_fallback_on_random_fail

    def fit(self, z,  
            weight_index = 0, 
            current_positions = None, 
            current_height = 0,
            current_value_streak = 0, 
            current_alternation=0, 
            parent_type="event", 
            force_event_test=True): #z is the dataset preprocessed
        if current_positions is None:
            self.tree = self.fit(z,  
                            weight_index, 
                            z.init_positions(weight_index),  
                            current_height, 
                            current_value_streak , 
                            current_alternation, 
                            parent_type, 
                            force_event_test)
            return self.tree
        current_samples = {i[0] for i in current_positions}
        r = {"probabilities": z.compute_probabilities(z.get_weights(current_samples, weight_index))}
        
        if (r["probabilities"]["probability"] == 1).mean() > 0:
            return r
        elif current_height >= self.max_height:
            return r
        elif z.get_weights(current_samples, weight_index).weight.sum() < self.min_node_samples:
            return r
        
        allow_value_test = True
        if force_event_test:
            allow_value_test =  False
        elif self.max_value_streak <= current_value_streak:
            allow_value_test =  False
        elif self.max_alternation <= current_alternation and parent_type == "event":
            allow_value_test =  False    

        if allow_value_test:
            value_loss, value_test, value_true, value_false, value_true_weight, value_false_weight = z.max_by_test("value", 
                                                                            current_positions,
                                                                            weight_index,
                                                                            self.value_loss, self.random_samples)
            if value_loss  == 0:
                allow_value_test = False
                if (self.random_samples is not None) and (not self.best_fallback_on_random_fail):
                    value_loss, value_test, value_true, value_false, value_true_weight, value_false_weight = z.max_by_test("value", 
                                                                            current_positions,
                                                                            weight_index,
                                                                            self.value_loss, None)
                    if value_loss > 0:
                        allow_value_test = True
         
        allow_event_test = True
        if self.max_alternation <= current_alternation and parent_type == "value":
            allow_event_test =  False

        if allow_value_test and self.alternate_if_possible and parent_type == "event":
            allow_event_test = False

        if allow_event_test:
            event_loss, event_test, event_true, event_false, event_true_weight, event_false_weight = z.max_by_test("event", 
                                                                            current_positions,
                                                                            weight_index,
                                                                            self.event_loss, self.random_samples)
            if event_loss  == 0:
                allow_event_test = False
                if (self.random_samples is not None) and (not self.best_fallback_on_random_fail):
                    event_loss, event_test, event_true, event_false, event_true_weight, event_false_weight = z.max_by_test("event", 
                                                                            current_positions,
                                                                            weight_index,
                                                                            self.event_loss, None)
                    if event_loss > 0:
                        allow_event_test = True    

        if not allow_value_test and not allow_event_test:
            return r

        if allow_value_test:
            next_value_streak = current_value_streak + 1
            next_parent_type = "value"
            r["test_type"] = "value"
            r["test"] = value_test
            r["true_samples"] = value_true_weight
            r["false_samples"] = value_false_weight
            next_true_positions = value_true
            next_false_positions = value_false
            next_force_event_test = False

        if allow_event_test and (( not allow_value_test or event_loss >= value_loss ) or (self.alternate_if_possible and parent_type == 'value') ):
            next_value_streak = 0
            next_parent_type = "event"
            r["test_type"] = "event"
            r["test"] = event_test
            r["true_samples"] = event_true_weight
            r["false_samples"] = event_false_weight
            next_true_positions = event_true
            if self.early_removal:
                next_false_positions = z.filter_out_last_positions(event_false)
            else:
                next_false_positions = event_false
            next_force_event_test = True    

        if parent_type != next_parent_type:
            next_alternation = current_alternation + 1
        else:
            next_alternation = current_alternation
        
        if len(next_true_positions) > 0:
            r[True] = self.fit(z, 
                               weight_index, 
                               next_true_positions, 
                               current_height + 1,
                               next_value_streak,
                               next_alternation,
                               next_parent_type,
                               False #the next_force_event_test is only needed for the false child below,
                               )
            
        if len(next_false_positions) > 0:
            r[False] = self.fit(z, 
                               weight_index, 
                               next_false_positions, 
                               current_height + 1,
                               next_value_streak,
                               next_alternation,
                               next_parent_type,
                               next_force_event_test,
                               )
            
        return r

    
    def predict(self, z, weight_index=0, current_subtrees =  None):
        if current_subtrees is None:
            excluded_positions = z.excluded_positions(weight_index)
            z.add_prediction_dataframe(weight_index)
            self.predict(
                z,
                weight_index,
                {p: self.tree for p in excluded_positions}   
            )
            excluded_samples = {i[0] for i in excluded_positions}
            return z.predictions[weight_index][[ i in excluded_samples for i in range(len(z.sample_df))]]
        
        if not current_subtrees:
            return
        
        prediction = z.predictions[weight_index]
        next_subtrees, last_positions = {}, {}
        current_positions = set(current_subtrees.keys())
        last_positions = z.filter_out_inside_positions(current_positions)

        for p in current_positions:
            move = None 
            next_position = p
            current_sample = p[0]
            current_subtree = current_subtrees[p]
            prob_dict = DataFrameHelper.df_column_to_dict(current_subtree["probabilities"],'probability')
            prediction.loc[current_sample].nodes.append({k:v for k,v in current_subtree.items() if k not in {True, False, "probabilities"} }| {"probabilities": prob_dict})
            prediction.loc[current_sample].positions.append((p[1], z.raw_x[p[0]][p[1]]))
            
            if "test" in current_subtree:
                keep = True
                test_type = current_subtree["test_type"]
                test = current_subtree["test"]
               
                if  test_type == "event": 
                    if p not in last_positions:
                        next_distances = z.select_next({p})
                        test_location = (*p, test[0])
                        if (test_location in next_distances.index)  and ( next_distances.loc[test_location].distance <= test[1]):
                            move = True     
                            next_position = (p[0], int(next_distances.loc[test_location].next_position))
                        else:
                            move = False
                    else:
                        move = False
                        prediction.loc[current_sample].end_of_sequence = len(prediction.loc[current_sample])
                        if self.early_stop:
                            keep = False
                elif test_type == "value":
                    if z.select_value({p}).iloc[0].value <= test:
                            move = True
                    else:
                            move = False    
            else:
                keep = False
            
            if keep and (move not in current_subtree):
                keep = False    

            if not keep:
                prediction.loc[current_sample].prediction = max(prob_dict, key=prob_dict.get)

            if keep: 
                next_subtrees[next_position] = current_subtree[move]

            if keep:
                prediction.loc[current_sample].path.append(move)

        self.predict(z, weight_index, next_subtrees)