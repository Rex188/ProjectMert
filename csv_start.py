import pandas as pd

# 读取数据
tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

# 选出 subset = 'small' 的部分
df_small = tracks[tracks[('set', 'subset')] == 'small']

# 选出需要的列
df_small_simple = df_small[[('track', 'genre_top')]].copy()

# 把 index 变成一列
df_small_simple = df_small_simple.reset_index()
df_small_simple.columns = ['track_id', 'genre_top']

# 保存为 CSV
df_small_simple.to_csv('fma_small_labels.csv', index=False)

print("✅ 成功保存为 fma_small_labels.csv")
