
# def
# default parameter can be defined
# function annotation for arguments type is followed by ':', for return type is followed by '->'
def fib(n: int = 4, is_print = True) -> int:
    a, b = 0, 1
    while a < n:
        a, b = b, a+b
    if (is_print):
        print(b)
    return b

if __name__ == "__main__":

    # if
    import random

    rand_gen = random.Random()
    x = rand_gen.randint(1, 3)
    if x == 1:
        print("1")
    elif x == 2:
        print("2")
    else:
        print("other")

    # for
    # not: The for statement can only support iterating over an sequence of elements.
    for i in range(5):
        if i == 3:
            continue
        elif i == 4:
            break
        else:
            print(i)

    # pass
    # note: The pass statement does nothing. It can be used when a statement is required syntactically but the program
    # requires no action
    class MyEmptyClass:
        pass



    fib(10, is_print=False)  # keyword arguments can be supplied with form similar to 'is_print=False'
    f = fib  # function can be assigned to a variable
    f(10)


    # we can use '/' and '*' to divide the arguments to the function to be pos_only, standard, or keyword_only
    def combined_example(pos_only:str, /, standard:str, *, kwd_only:str):
        print(pos_only, standard, kwd_only)


    combined_example(1, 2, kwd_only=3)

    # arbitrary argument lists
    def concat(*args, sep="/"):
        return sep.join(args)


    print(concat("a", "b", "c"))

    # unpack arguments list, we can add '*' before arguments to unpack the tuple arguments
    args = [3, 6]
    print(list(range(*args)))

    # lambda expression
    f = lambda x: x+1
    print(f(1))