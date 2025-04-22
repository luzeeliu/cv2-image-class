# longtail_config.py

def generate_longtail_ratios(num_classes=15, max_ratio=1.0, min_ratio=0.1):
    """
Generates arithmetic decreasing class sampling ratios to simulate long-tail distribution.
For example, class_0: 1.0, class_14: 0.1.

Args:
num_classes: total number of classes (default is 15)
max_ratio: maximum class sampling ratio (default is 1.0)
min_ratio: minimum class sampling ratio (default is 0.1)

Returns:
List[float]: sampling ratio of each class (from most to least)
    """
    step = (max_ratio - min_ratio) / (num_classes - 1)
    return [max_ratio - i * step for i in range(num_classes)]


if __name__ == "__main__":
    ratios = generate_longtail_ratios()
    print("🔢 Long-tail sample ratios:", [round(r, 3) for r in ratios])
