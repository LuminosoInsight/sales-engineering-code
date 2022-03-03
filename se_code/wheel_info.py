
def main():
    with open('wheel_info.txt') as f:
        line = f.readline()
        while line:
            print(line.rstrip())
            line = f.readline()


if __name__ == '__main__':
    main()
