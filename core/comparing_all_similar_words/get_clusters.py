import pandas as pd
df = pd.read_csv("result/clusters.csv")
cross = df[df.n_pages >= 2].sort_values(["cluster", "page"])
print(cross.groupby("cluster")["page"].unique())