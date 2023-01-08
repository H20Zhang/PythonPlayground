
if __name__ == "__main__":

    # Numeric
    print(1 + 1)  # add 2
    print(5 / 7)  # classic division returns a float
    print(5 // 7)  # floor division disards the fractional part
    print(5 % 7)  # the % returns the reminder
    print(5 ** 2)  # calculate power

    # String
    print('abcd\n')  # normal string
    print(r'abcd')  # raw string, which starts with 'r'
    print("""\
    abcd
    1234
    5678
    """)  # multi-line of string are includes with """""""
    print(3 * 'a')  # string repetition
    print('a' + 'b')  # string concatenation
    print('123456'[0])  # string indexing
    print('123456'[0:2])  # string slicing
    #python string are immutable


    # List, list are mutable
    square = [1, 4, 9, 16, 25]
    print(square[0])  # list indexing
    print(square[1:2])  # list slicing
    print(square[-3:])  # list slicing using negative index
    print(square[:])  # all slice operations return a new list, thus this clause returns a shallow copy
    print(square + [36, 49, 64])  # list concatenation
    square[1:3] = [10, 10]; print(square)  # list can be modified
    nested_list = [1, 2, 3, [1, 2, 3]]; print(nested_list)  # list can be nested
    square.append(81); print(square)  # append an element at the end of the list
    square.pop(); print(square)  # pop the element at the end of the list
    square.extend([81, 100]); print(square)  # list extension
    square.insert(1, 2); print(square)  # insert an element at given index
    square.remove(2); print(square)  # remove the first element with given number
    square.reverse(); print(square)  # reverse the list
    square.sort(); print(square)  # sort the list in place
    square = list(map(lambda x: x ** 2, range(10))); print(square)  # list comprehensions
    square = [x**2 for x in range(10)]; print(square)  # another way of list comprehensions
    del square[0]; print(square) # delete an element from the list given index
    for i, v in enumerate(square):  # we can get the index and value using enumerate
        print(i, v)
    for k, v in zip(square, square):  # we can use zip to zip two list
        print(k, v)
    for i in reversed(square):  # we loop a list reversely
        print(i)
    for i in sorted(square):  #  loop through a list in ordered way
        print(i)

    # tuple, tuples are similar to list but are immutable
    t = (1, 2, 3)
    a, b, c = t  # tuple unpacking
    print(a, b, c)
    # clause like 't[1] = 0' will throw error



    # set
    basket = {'1', '2'}
    print(basket)   # print the set
    print('1' in basket)  # check if an element is in the set

    set1 = {1, 2}
    set2 = {2, 3}
    print(set1 - set2)  # set difference
    print(set1 | set2)  # set union
    print(set1 & set2)  # set intersection
    print(set1 ^ set2)  # the complement of set intersection

    set3 = {x for x in 'abcdefg' if x not in 'abc'}  # list comphrehension for set
    print(set3)

    # dictionary
    dic1 = {1: 'a', 2: 'b'}
    print(dic1[1])  # dictionary indexing
    print(list(dic1))  # create list from key of dictionary
    print(1 in dic1)  # find if the dictionary contains the key
    print({x: x**2 for x in (2, 4, 5)})  # list comprehension for dictionary
    for k, v in dic1.items():  # we can iterate through key value of the dictionary
        print(k, v)