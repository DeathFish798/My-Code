#!/usr/bin/python3
list = []
while True:
    try:
        list.append(int(input()))
    except:
        break
max = 0
for i in list:
    ch = 0
    for j in list:
        if j == i:
            ch += 1
    if ch > max:
        max = ch
print(max)