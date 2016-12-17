from data_proc import build_graph

W = build_graph('movielens.tsv')
print W[0, :]
