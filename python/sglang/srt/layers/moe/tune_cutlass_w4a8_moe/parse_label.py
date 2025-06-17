def parse_label(label_value):
    """
    从label值反向解析出a, b, c, d, e, f参数
    
    原公式：
    label_value = int((a/64) * 1e5 + (b/64) * 1e4 + (c/64) * 1e3 + (d/1) * 1e2 + (e/1) * 1e1 + (f/1) * 1e0)
    
    参数:
        label_value: 输入的label值
        
    返回:
        (a, b, c, d, e, f) 元组
    """
    # 分解label_value的各个部分
    # 获取a部分 (1e5位)
    a_part = label_value // 100000
    a = a_part * 64
    
    # 获取b部分 (1e4位)
    b_part = (label_value % 100000) // 10000
    b = b_part * 64
    
    # 获取c部分 (1e3位)
    c_part = (label_value % 10000) // 1000
    c = c_part * 64
    
    # 获取d部分 (1e2位)
    d = (label_value % 1000) // 100
    
    # 获取e部分 (1e1位)
    e = (label_value % 100) // 10
    
    # 获取f部分 (1e0位)
    f = label_value % 10
    
    return (a, b, c, d, e, f)

# 示例用法
if __name__ == "__main__":
    # 测试几个示例label
    test_labels = [113211]
    
    for label in test_labels:
        a, b, c, d, e, f = parse_label(label)
        print(f"Label: {label} => a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
        
        # 验证反向计算是否正确
        calculated_label = int((a/64) * 1e5 + (b/64) * 1e4 + (c/64) * 1e3 + (d/1) * 1e2 + (e/1) * 1e1 + (f/1) * 1e0)
        assert calculated_label == label, f"验证失败: {calculated_label} != {label}"