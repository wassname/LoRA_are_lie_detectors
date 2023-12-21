import pandas as pd 
import numpy as np

def filter_ds_to_known(ds1, verbose=True):
    """filter the dataset to only those where the model knows the answer"""
    
    # first get the rows where it answered the question correctly
    df = ds2df(ds1).rename(columns=lambda x: x.replace('_base', ''))
    d = df.query('sys_instr_name=="truth"').set_index("example_i")
    m1 = (d.binary_ans>0.5)==d.label_true
    known_indices = d[m1].index
    known_rows = df['example_i'].isin(known_indices)
    known_rows_i = df[known_rows].index
    
    if verbose: print(f"select rows are {m1.mean():2.2%} based on knowledge")
    return ds1.select(known_rows_i)


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
    
    # json.loads(dss[0].info.description)['f'] # doesn't work when concat

    if cols is None:
        r = ds[0]
        # get all the columns that not large lists or arrays
        cols = [k for k,v in r.items() if (isinstance(v, np.ndarray) and v.size<2) or not isinstance(v, (list, np.ndarray))]
    ds = ds.with_format('numpy')
    df = ds.select_columns(cols)
    df = pd.DataFrame([rows_item(r) for r in df])
    return df

def qc_ds(ds):
    df = ds2df(ds.with_format('numpy')).rename(columns=lambda x: x.replace('_base', ''))
    df['ans'] = df['binary_ans'] >0.5


    df['label_instructed'] = df['label_true'] ^ df['instructed_to_lie']


    # check llm accuracy
    d = df.query('instructed_to_lie==False')
    acc = (d.label_instructed==d['ans']).mean()
    assert np.isfinite(acc)
    print(f"\tacc    =\t{acc:2.2%} [N={len(d)}] - when the model is not lying... we get this task acc")
    assert acc>0.3, f"model cannot solve task acc={acc}"

    # check LLM lie freq
    d = df.query('instructed_to_lie==True')
    acc = (d.label_instructed==d['ans']).mean()
    assert np.isfinite(acc)
    print(f"\tlie_acc=\t{acc:2.2%} [N={len(d)}] - when the model tries to lie... we get this acc")
    assert acc>0.01, f"no known lies acc={acc}"

    # check LLM lie freq
    ds_known = filter_ds_to_known(ds, verbose=False)
    df_known = ds2df(ds_known).rename(columns=lambda x: x.replace('_base', ''))
    df_known['ans'] = df_known['binary_ans'] >0.5
    df_known['label_instructed'] = df_known['label_true'] ^ df_known['instructed_to_lie']
    d = df_known.query('instructed_to_lie==True')
    acc = (d.label_instructed==d['ans']).mean()
    assert np.isfinite(acc)
    print(f"\tknown_lie_acc=\t{acc:2.2%} [N={len(d)}] - when the model tries to lie and knows the answer... we get this acc")
    assert acc>0.01, f"no known lies={acc}"

    # check choice coverage
    mean_prob = ds['choice_probs_adapt'].mean()
    print(f"\tchoice_cov=\t{mean_prob:2.2%} - Our choices accounted for a mean probability of this")
    assert mean_prob>0.1, "neither of the available choice very likely :(, try debuging your templates. Check: using the correct prompt, the whitespace is correct, the correct eos_tokens (if any)"
