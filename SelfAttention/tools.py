if __name__ == '__main__':
    file = "../Data/yoochoose/yoochoose-data/yoochoose-clicks.dat"
    file_out = "../Data/yoochoose/yoochoose-data/yoochoose-clicks-small.dat"
    content = []
    with open(file, 'r') as f:
        for line in f:
            content.append(line)
    print(len(content))

    for i in range(100):
        print(content[i])

    small_index = len(content) // 8
    # small_index = 100
    with open(file_out, 'w') as f:
        for line in content[-small_index:]:
            f.write(line)