import re
import sys

def main():
    if len(sys.argv) != 3:
        print("用法: python log_parser.py <输入文件> <输出文件>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 正则表达式模式匹配目标行
    pattern = r'batch_size:\s*(\d+)\s*label:\s*(\d+)\s*test_time:\s*([\d.]+)\s*ms'
    
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                batch_size = int(match.group(1))
                label = int(match.group(2))
                test_time = float(match.group(3))
                results.append((batch_size, label, test_time))
    
    # 按batch_size、label、test_time排序
    sorted_results = sorted(results, key=lambda x: (x[0], x[2], x[1]))
    
    # 生成Markdown表格
    markdown = "| batch_size | label | test_time (ms) |\n"
    markdown += "|------------|-------|----------------|\n"
    for batch_size, label, test_time in sorted_results:
        markdown += f"| {batch_size} | {label} | {test_time:.8f} |\n"
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"已成功生成表格到 {output_file}")

if __name__ == "__main__":
    main()