import sys
from collections import defaultdict, Counter

def parse_markdown_table(markdown_content):
    """解析Markdown表格内容"""
    lines = markdown_content.strip().split('\n')
    results = []
    # 跳过表头和分隔行
    for line in lines[2:]:
        parts = [part.strip() for part in line.split('|') if part.strip()]
        if len(parts) == 3:
            try:
                batch_size = int(parts[0])
                label = parts[1] # label保持为字符串
                test_time = float(parts[2])
                results.append((batch_size, label, test_time))
            except ValueError as e:
                print(f"Skipping line due to parsing error: {line} - {e}")
    return results

def find_top_labels_per_batch_presorted(parsed_data, top_n=5):
    """为每个batch_size找出top_n的label (假设输入数据已按test_time排序)"""
    batch_groups = defaultdict(list)
    for batch_size, label, test_time in parsed_data:
        batch_groups[batch_size].append(label) # 只需要label，因为已经排序
    
    top_labels_by_batch = defaultdict(set)
    for batch_size, labels_in_order in batch_groups.items():
        # 直接取前top_n个，因为数据已排序
        top_labels_by_batch[batch_size] = set(labels_in_order[:top_n])
    return top_labels_by_batch

def find_consistent_labels(top_labels_by_batch):
    """找出在所有batch_size中都存在的top label"""
    if not top_labels_by_batch:
        return set()
    
    num_batch_sizes = len(top_labels_by_batch)
    label_counts = Counter()
    for batch_size in top_labels_by_batch:
        for label in top_labels_by_batch[batch_size]:
            label_counts[label] += 1
            
    consistent_labels = {label for label, count in label_counts.items() if count == num_batch_sizes}
    return consistent_labels

def main():
    if len(sys.argv) != 3:
        print("Usage: python select_best_config.py <file_path> <top_n>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    top_n = int(sys.argv[2])

    try:
        with open(file_path, 'r') as f:
            markdown_input = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()

    parsed_data = parse_markdown_table(markdown_input)
    if not parsed_data:
        print("No data parsed. Exiting.")
        exit()

    # print("Parsed Data:")
    # for item in parsed_data:
        # print(item)
    # print("-"*20)

    TOP_N_FILTER = top_n
    # 使用简化后的函数
    top_labels_per_batch_size = find_top_labels_per_batch_presorted(parsed_data, top_n=TOP_N_FILTER)

    # print(f"Top {TOP_N_FILTER} labels per batch size (assuming pre-sorted input):")
    # for bs, labels in top_labels_per_batch_size.items():
    #     print(f"Batch Size {bs}: {labels}")
    # print("-"*20)

    consistent_good_labels = find_consistent_labels(top_labels_per_batch_size)

    if consistent_good_labels:
        print(f"Labels that are in the top {TOP_N_FILTER} for ALL batch sizes:")
        for label in consistent_good_labels:
            print(label)
    else:
        print(f"No labels found that are in the top {TOP_N_FILTER} for all batch sizes.")




# --- 主程序 ---
if __name__ == "__main__":
    main()