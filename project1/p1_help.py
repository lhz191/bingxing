from openpyxl import Workbook

# 创建一个新的工作簿
wb = Workbook()

# 激活默认的工作表
ws = wb.active

# 指定要写入的文件路径
file_path = "p1_data.xlsx"

# 设置矩阵大小
n = 200

# 写入数据到工作表
for i in range(n):
    for j in range(n):
        # 计算值为 i + j
        value = i + j
        # 写入到单元格中
        ws.cell(row=i+1, column=j+1, value=value)

# 保存工作簿到文件
wb.save(filename=file_path)
