import logging




def ngram_distance(items1: list, items2: list, n: int = 1):
    if len(items1) < n or len(items2) < n:
        return None
    joined_items1 = [' '.join(items1[i:i + n]) for i in range(len(items1) - n + 1)]
    joined_items2 = [' '.join(items2[i:i + n]) for i in range(len(items2) - n + 1)]
    common_items = set(joined_items1) & set(joined_items2)
    return len(joined_items1) + len(joined_items2) - 2 * len(common_items)

def knn():
    pass


if __name__ == '__main__':
    command1 = 'cat testfile > output.txt'
    command2 = 'cat testfile2 > output.txt'
    print(ngram_distance(command1.split(' '), command2.split(), 3))
