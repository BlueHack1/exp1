import json



def delete_lines_range_from_jsonl(input_file, output_file, start_line, end_line):
    """
    删除 jsonl 文件中的指定行区间 [start_line, end_line]（从 0 开始计数）
    :param input_file: 输入 jsonl 文件路径
    :param output_file: 输出 jsonl 文件路径
    :param start_line: 起始行号 (包含)
    :param end_line: 结束行号 (包含)
    """
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if not (start_line <= idx <= end_line):
                f_out.write(line)

# 使用示例：删除第 10 到 20 行（包含 10 和 20）
delete_lines_range_from_jsonl(
    input_file="yelp_academic_dataset_business.json",
    output_file="yelp_academic_dataset_business_new.jsonl",
    start_line=100000,
    end_line=150000
)
