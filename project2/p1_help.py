import pandas as pd

# 定义数组大小
n = 262144

# 生成数组
a = [i for i in range(n)]

# 将数组写入Excel文件
pd.DataFrame(a).to_excel('project2_data.xlsx', header=False, index=False)
