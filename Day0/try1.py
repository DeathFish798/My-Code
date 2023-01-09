#list and tup
a = [3,'r',"jio"]
b = [3,2,3,4,5]
del(a[1])

a.insert(1,'k')
print(a)

a.reverse()
print(a)

a.pop()
a.append(9)
print(a)

a.extend(b)
print(a)

c = a.index(3)
print(c)

t = a.count(3)
print(t)

tuple(a)
list(a)
print(a)
print("\\\\\\\\\\\\\\")

#dict
dict={'name':'QQ','age':18,'height':182}
print(dict)

for k,v in dict.items():
    print(k,v)
for key in dict.keys():
    print(key)
for value in dict.values():
    print(value)

dict.popitem()
print(dict)

#iter
ia = iter(a)
ib = iter(b)
print(a,b)
for xa in ia:
    print(xa)