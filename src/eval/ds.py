import pandas as pd 
import numpy as np

# def filter_ds_to_known(ds1, verbose=True):
#     """filter the dataset to only those where the model knows the answer"""
    
#     # first get the rows where it answered the question correctly
#     df = ds2df(ds1).rename(columns=lambda x: x.replace('_base', ''))
#     d = df.query('sys_instr_name=="truth"').set_index("example_i")
#     m1 = (d.binary_ans>0.5)==d.label_true
#     known_indices = d[m1].index
#     known_rows = df['example_i'].isin(known_indices)
#     known_rows_i = df[known_rows].index
    
#     if verbose: print(f"select rows are {m1.mean():2.2%} based on knowledge")
#     return ds1.select(known_rows_i)

def filter_df_to_known(df, verbose=True):
    """filter the dataset to only those where the model knows the answer"""
    
    # first get the rows where it answered the question correctly
    d = df.query('sys_instr_name=="truth"').set_index("example_i")
    m1 = (d.binary_ans>0.5)==d.label_true
    known_indices = d[m1].index
    known_rows = df['example_i'].isin(known_indices).copy()
    if verbose: print(f"select rows are {m1.mean():2.2%} based on knowledge")
    return df[known_rows]

def rows_item(row):
    """
    transform a row by turning single dim arrays into items
    """
    for k,x in row.items():
        if isinstance(x, np.ndarray) and (x.ndim==0 or (x.ndim==1 and len(x)==1)):
            row[k]=x.item()
        if isinstance(x, list) and len(x)==1:
            row[k]=x[0]
    return row


def ds2df(ds, cols=None):
    """one of our custom datasets into a dataframe
    
    dropping the large arrays and lists"""
    ds = ds.with_format('numpy')
    
    # json.loads(dss[0].info.description)['f'] # doesn't work when concat

    if cols is None:
        r = ds[0]
        # get all the columns that not large lists or arrays
        cols = [k for k,v in r.items() if (isinstance(v, np.ndarray) and v.size<2) or not isinstance(v, (list, np.ndarray))]
    ds = ds.with_format('numpy')
    df = ds.select_columns(cols)
    df = pd.DataFrame([rows_item(r) for r in df])

    if 'choice_probs_adapt' in ds.column_names:
        df['choice_probs_adapt'] = np.sum(ds['choice_probs_adapt'], 1)
        df['ans_adapt'] = df['binary_ans_adapt'] >0.5
    df['choice_probs_base'] = np.sum(ds['choice_probs_base'], 1)
    df['ans_base'] = df['binary_ans_base'] >0.5
    df['label_instructed'] = df['label_true_base'] ^ df['instructed_to_lie_base']
    return df.copy()


def qc_dsdf(df):

    res = {}

    print(f"\tbalance=\t{df['label_true'].mean():2.2%} [N={len(df)}]")
    res['balance'] = df['label_true'].mean()
    res['N'] = len(df)

    d = df.query('instructed_to_lie==False')
    if len(d):
        acc = (d.label_instructed==d['ans']).mean()
        # assert np.isfinite(acc)
        print(f"\tacc    =\t{acc:2.2%} [N={len(d)}]      - when the model is not lying... we get this task acc")
        if acc<=0.3:
            print(f"WARNING: model cannot solve task acc={acc}")
        res['acc'] = acc

    # check LLM lie freq
    d = df.query('instructed_to_lie==True')
    if len(d):
        acc = (d.label_instructed==d['ans']).mean()
        # assert np.isfinite(acc)
        print(f"\tlie_acc=\t{acc:2.2%} [N={len(d)}]      - when the model tries to lie... we get this acc")
        if acc<=0.01:
            print(f"WARNING: no known lies {acc}")
        res['lie_acc'] = acc

    # check LLM lie freq
    df_known = filter_df_to_known(df, verbose=False)
    d = df_known.query('instructed_to_lie==True')
    if len(d):
        acc = (d.label_instructed==d['ans']).mean()
        # assert np.isfinite(acc)
        print(f"\tknown_lie_acc=\t{acc:2.2%} [N={len(d)}]      - when the model tries to lie and knows the answer... we get this acc")
        if acc<=0.01:
            print(f"WARNING: no known lies {acc}")
        # assert acc>0.01, f"no known lies={acc}"
        res['known_lie_acc'] = acc

    # check choice coverage
    mean_prob = df['choice_probs'].mean()
    print(f"\tchoice_cov=\t{mean_prob:2.2%}             - Our choices accounted for a mean probability of this")
    assert mean_prob>0.1, "neither of the available choice very likely {mean_prob:2.2%} :(, try debuging your templates. Check: using the correct prompt, the whitespace is correct, the correct eos_tokens (if any)"
    res['tchoice_cov'] = mean_prob
    return res

def qc_ds(ds):
    df = ds2df(ds)
    df = df.rename(columns=lambda x: x.replace('_base', '')).copy()
    # check llm accuracy
    print('with base model')
    res_b = qc_dsdf(df)
    if 'label_true_adapt' in ds.column_names:
        print('with adapter')
        df = ds2df(ds)
        df = df.rename(columns=lambda x: x.replace('_adapt', '')).copy()
        res_b = qc_dsdf(df)
        return res_b, res_b
    return res_b


