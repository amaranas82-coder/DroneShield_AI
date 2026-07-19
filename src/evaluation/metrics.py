import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    average_precision_score,
    confusion_matrix
)

class MetricsCalculator:
    """حساب مقاييس التقييم المتقدمة."""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
        """حساب جميع المقاييس."""
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['mAP'] = average_precision_score(y_true, y_pred_proba)
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        total = cm.sum()
        metrics['accuracy'] = float((cm[0, 0] + cm[1, 1]) / total)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: dict):
        """طباعة المقاييس بشكل منسق."""
        print("\n" + "="*60)
        print("Performance Metrics")
        print("="*60)
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        
        if 'mAP' in metrics:
            print(f"mAP:        {metrics['mAP']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
        print("="*60)

if __name__ == "__main__":
    print("MetricsCalculator module ready")
