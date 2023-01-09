#!/usr/bin/python3
A = eval(input())
B = eval(input())

if len(A) != len(B):
    print("The length is not same")
else:
    sum = 0
    moda = 0
    modb = 0
    for i in range(len(A)):
        sum = sum + A[i] * B[i]
    for i in A:
        moda = moda + i ** 2
    for i in B:
        modb = modb + i ** 2
    moda = moda ** 0.5
    modb = modb ** 0.5

    if moda == 0 and modb == 0:
        cos = 1
    elif moda == 0 or modb == 0:
        cos = 0
    else:
        cos = sum / moda / modb
    print("{:.6f}".format(cos))
