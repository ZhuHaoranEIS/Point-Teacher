import random
from random import randint

range_city = [(1, 1051100), (1051101, 1741600), (1741601, 2491800), (2491801, 3392400), (3392401, 4674000), (4674001, 5187000), (5187001, 7905600)]

range_countryside = [(1, 925500), (925501, 1107100), (1107101, 2211800), (2211801, 3661000), (3661001, 4903700), (4903701, 5833400)]

random.seed(66)

while True:
    generate_city = []
    has = [False, False, False, False, False, False, False]
    for i in range(5):
        new_data = random.randint(1, 7905600)
        generate_city.append(new_data)

    # decide
    flag_exit = True
    for i in range(len(generate_city)):
        data_now = generate_city[i]
        for j in range(len(range_city)):
            if data_now >= range_city[j][0] and data_now <= range_city[j][1]:
                if has[j] != True:
                    has[j] = True
                else:
                    flag_exit = False
                break
        if flag_exit == False:
            break
    if flag_exit == True:
        print(has)
        print(sorted(generate_city))
        break

while True:
    generate_city = []
    has = [False, False, False, False, False, False]
    for i in range(4):
        new_data = random.randint(1, 5833400)
        generate_city.append(new_data)

    # decide
    flag_exit = True
    for i in range(len(generate_city)):
        data_now = generate_city[i]
        for j in range(len(range_countryside)):
            if data_now >= range_countryside[j][0] and data_now <= range_countryside[j][1]:
                if has[j] != True:
                    has[j] = True
                else:
                    flag_exit = False
                break
        if flag_exit == False:
            break
    if flag_exit == True:
        print(has)
        print(sorted(generate_city))
        break

        

    
