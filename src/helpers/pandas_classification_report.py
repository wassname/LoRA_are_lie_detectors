"""
@url: https://gist.github.com/wassname/f3cbdc14f379ba9ec2acfafe5c1db592
"""
import pandas as pd
import sklearn.metrics
import numpy as np

def classification_report(*args, **kwargs):
    """
    
    Usage
    ```py
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    classification_report(y, y_pred.argmax(-1), target_names=[0, 1, 2])
    ```
    
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |        0.23 |     0.05 |       0.08 |      3111 |
    | 1            |        0.76 |     0.86 |       0.8  |     14344 |
    | 2            |        0.21 |     0.25 |       0.23 |      2577 |
    | accuracy     |        0.65 |     0.65 |       0.65 |     20032 |
    | macro avg    |        0.4  |     0.39 |       0.37 |     20032 |
    | weighted avg |        0.6  |     0.65 |       0.62 |     20032 |
    """
    
    out_df = pd.DataFrame(sklearn.metrics.classification_report(*args, **kwargs, output_dict=True)).T
    # Order cols
    out_df[["precision","recall","f1-score","support"]]  
    # Round
    out_df[["precision","recall","f1-score"]]= out_df[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    out_df[["support"]]= out_df[["support"]].apply(lambda x: x.astype(int))
    # Add suport to avg
    out_df.loc['accuracy', 'support'] = out_df.loc['weighted avg', 'support']
    
    out_df = out_df.style.set_caption("classification_report")
    return out_df


def confusion_matrix(*args, target_names, **kwargs):
    """
    Confusion matrix to pandas dataframe
    
    Usage
    ```
    target_names=['cls_down', 'cls_end', 'cls_up']
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    cm2df(y_true, y_pred, target_names=target_names, normalize='true')
    ```
    
    
    |          |   cls_down |   cls_end |   cls_up |
    |:---------|-----------:|----------:|---------:|
    | cls_down |      0.051 |     0.699 |    0.249 |
    | cls_end  |      0.029 |     0.857 |    0.114 |
    | cls_up   |      0.048 |     0.699 |    0.253 |
    """

    cm = sklearn.metrics.confusion_matrix(*args, **kwargs)
    df = pd.DataFrame(cm, columns=target_names, index=target_names)
    df.index.name = 'Labels'
    df.columns.name = 'Pred'
    df = df.style.set_caption("confusion_matrix")
    return df
