#integer and string

x = 6
m = 9
y="4"
z="8"

print(x+m)
print(y+z)
print(type(y))

#typecasting
a=int(y)
print(a+m)

#list operations

l1=['hi',2,4,6,'hello',[1,2,3,'sakshi']]
l2= l1[5][2:]
print(l2)
length= len(l2)
print(length)

#append
l1.append(['tej',9])
print(l1)

#extend
l1.extend([8,7,6])
print(l1)

#tuples
t1=(1,'sakshi',2,3)
print(t1)

#dictionaries
d1={ 'Name': 'Sakshi', 'Roll.no':'26', 'Age':'20'}
print(d1)

#operations on tuples
print(t1[1])

t2=(45,44,68)
print(t2)





