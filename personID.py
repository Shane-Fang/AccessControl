import csv

'''
    處理ID.csv檔所需的功能
    ID.csv檔是拿來儲存流水號ID和人名的
    因為模型輸出結果會是一組陣列，裡面的值為輸入的影像人臉判別機率
    所以用這組陣列去跟ID.csv檔去做搭配，就可以輸出辨別結果
'''


def writeID(name: str):
    # 寫入人名和ID，
    if checkNameExist(name):
        with open("ID.csv", "r", encoding='utf8') as f:
            ID = len(f.readlines()) - 1

        if ID == 0:
            with open("ID.csv", "w", newline='') as f:
                fieldnames = ['ID', 'name']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'ID': ID, 'name': name})
        else:
            with open("ID.csv", "a", newline='') as f:
                fieldnames = ['ID', 'name']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'ID': ID, 'name': name})


def getPersonByID(id: int):
    # 用ID檢索人名，存在則返回string
    with open("ID.csv", "r", encoding='utf8') as f:
        rows = csv.DictReader(f)
        for row in rows:
            if row['ID'] == str(id):
                return row['name']
        print('The person is not exist')


def getIDByName(name: str):
    # 用人名檢索ID，存在則返回int
    with open("ID.csv", "r", encoding='utf8') as f:
        rows = csv.DictReader(f)
        for row in rows:
            if row['name'] == name:
                return int(row['ID'])
        print('The person is not exist')


def getPersonNum() -> int:
    # 獲得ID.csv檔人數，返回int
    with open("ID.csv", "r", encoding='utf8') as f:
        return len(f.readlines()) - 1


def checkNameExist(name: str) -> bool:
    # 檢查人是否存在，返回boolean
    with open("ID.csv", "r", encoding='utf8') as f:
        rows = csv.DictReader(f)
        for row in rows:
            if row['name'] == name:
                print('The person is exist')
                return False
    return True


def getPersonAndID():
    # 獲得ID.csv檔所有資料，返回string, int
    persons = []
    ID = []
    with open("ID.csv", "r", encoding='utf8') as f:
        rows = csv.DictReader(f)
        for row in rows:
            persons.append(row['name'])
            ID.append(row['ID'])
    return persons, ID


if __name__ == '__main__':
    print(getPersonAndID())
