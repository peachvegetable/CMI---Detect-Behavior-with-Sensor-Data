"""
Competition metric for CMI 2025
"""

import pandas as pd
from sklearn.metrics import f1_score

class CompetitionMetric:
    def __init__(self):
        self.bfrb_gestures = [
            'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
            'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - pinch skin',
            'Neck - scratch', 'Cheek - pinch skin'
        ]
    
    def calculate_hierarchical_f1(self, true_df, pred_df):
        """
        Calculate the hierarchical F1 score as per competition rules.
        Competition metric = (binary_f1 + multiclass_f1) / 2
        """
        y_true = true_df['gesture'].values
        y_pred = pred_df['gesture'].values
        
        # Binary classification: BFRB vs non-BFRB
        y_true_binary = [1 if g in self.bfrb_gestures else 0 for g in y_true]
        y_pred_binary = [1 if g in self.bfrb_gestures else 0 for g in y_pred]
        binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
        
        # Multiclass: Map non-BFRB to single class
        y_true_multi = [g if g in self.bfrb_gestures else 'non_target' for g in y_true]
        y_pred_multi = [g if g in self.bfrb_gestures else 'non_target' for g in y_pred]
        multi_f1 = f1_score(y_true_multi, y_pred_multi, average='macro')
        
        # Competition score is average of both
        competition_score = (binary_f1 + multi_f1) / 2
        
        return competition_score