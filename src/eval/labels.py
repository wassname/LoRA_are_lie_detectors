def ds2label_model_obey(ds):
    """extract label from hs dataset, for cases where model obeys instructions (wether to lie or not)"""
    label_instructed = ds["label_true_base"] ^ ds["instructed_to_lie_base"]
    ans = ds["binary_ans_base"] > 0.5
    labels_untruth = label_instructed == ans
    return labels_untruth


def ds2label_model_truth(ds):
    ans = ds["binary_ans_base"] > 0.5
    labels_true_ans = ds["label_true_base"] == ans
    return labels_true_ans
