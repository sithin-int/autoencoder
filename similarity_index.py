import pandas as pd
from scipy.optimize import linear_sum_assignment

#un-ordered similarity metric

def jaccard(df1, df2, k, **kwargs):

    N = len(df1) # or len(df2)

    assert set(["feature", "rank"]).issubset(df1.columns) and set(["feature", "rank"]).issubset(df2.columns), "missing columns - feature, rank"
    assert k<=N, f"k should be <={N}"

    f1 = df1.sort_values(by="rank")["feature"].to_list()
    f2 = df2.sort_values(by="rank")["feature"].to_list()

    f1_k, f2_k = set(f1[:k]), set(f2[:k])

    return len(f1_k & f2_k)/len(f1_k | f2_k)

def dice(df1, df2, k, **kwargs):

    N = len(df1) # or len(df2)

    assert set(["feature", "rank"]).issubset(df1.columns) and set(["feature", "rank"]).issubset(df2.columns), "missing columns - feature, rank"
    assert k<=N, f"k should be <={N}"

    f1 = df1.sort_values(by="rank")["feature"].to_list()
    f2 = df2.sort_values(by="rank")["feature"].to_list()
    
    f1_k, f2_k = set(f1[:k]), set(f2[:k])

    return (2 * len(f1_k & f2_k))/(len(f1_k)+len(f2_k))

def kuncheva(df1, df2, k, **kwargs):

    # for kuncheva index boundaries to work you have to make sure that features in df1 and df2 are the same

    N = len(df1) # or len(df2)

    assert set(["feature", "rank"]).issubset(df1.columns) and set(["feature", "rank"]).issubset(df2.columns), "missing columns - feature, rank"
    assert k<=N, f"k should be <={N}"
    
    if k==0 or k==N: #not defined
        return 0.0

    f1 = df1.sort_values(by="rank")["feature"].to_list()
    f2 = df2.sort_values(by="rank")["feature"].to_list()

    f1_k, f2_k = set(f1[:k]), set(f2[:k])

    r = len(f1_k & f2_k)

    return (r*N - (k**2))/(k * (N-k))

def mwm(df1, df2, feats_df, k, **kwargs):

    assert set(["feature", "rank"]).issubset(df1.columns) and set(["feature", "rank"]).issubset(df2.columns), "missing columns - feature, rank"

    f1 = df1.sort_values(by="rank")["feature"].to_list()
    f2 = df2.sort_values(by="rank")["feature"].to_list()

    f1_k, f2_k = f1[:k], f2[:k]

    f = list(set(f1_k) | set(f2_k))
    corr_matrix = feats_df[f].corr(method='spearman').abs()
    cost_matrix = corr_matrix.loc[f1_k, f2_k]
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return cost_matrix.values[row_ind, col_ind].sum()/k

    
# ordered similarity metric

def global_spearman(df1, df2, **kwargs):

    assert set(["feature", "rank"]).issubset(df1.columns) and set(["feature", "rank"]).issubset(df2.columns), "missing columns - feature, rank"

    r1 = df1.sort_values(by="feature")["rank"].to_numpy()
    r2 = df2.sort_values(by="feature")["rank"].to_numpy()

    N = len(r1) # or len(r2)
    
    return 1 - ((6*(r1-r2)**2).sum()/(N*(N**2 - 1)))

    