
list = [1, 2, 3, 4]

print(list)
index_to_change = 0
for index, number in enumerate(list):
    if number == 2:
        index_to_change = index
        break

list[index_to_change] = 10

print(list)
