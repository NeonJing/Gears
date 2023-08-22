import pandas as pd

def modify_column_names(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将列名修改为指定的名字
    df.rename(columns={'document': 'prompt', 'summary': 'completion'}, inplace=True)

    # 将修改后的数据保存为新的CSV文件
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file_path = "E:\desktop\gears\AutoReview_summarized_train.csv"   # 输入CSV文件路径
    output_file_path = "E:\desktop\gears\example.csv"                # 输出CSV文件路径

    modify_column_names(input_file_path, output_file_path)
    print("已生成修改后的CSV文件:", output_file_path)
