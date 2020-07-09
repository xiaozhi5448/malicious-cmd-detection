



if __name__ == '__main__':

    def armstrong_number(a, b):
        results = []
        if 0 <= a <= b <= 999:
            for num in range(a, b + 1):
                num1 = num // 100
                tmp = num % 100
                num2 = tmp // 10
                num3 = tmp % 10
                if pow(num1, 3) + pow(num2, 3) + pow(num3, 3) == num:
                    results.append(num)
        return results


    print(armstrong_number(100, 0))
    print(armstrong_number(0, 100))
    print(armstrong_number(0, 999))
    print(armstrong_number(50, 100))
    print(armstrong_number(0, 0))
    print(armstrong_number(371, 371))