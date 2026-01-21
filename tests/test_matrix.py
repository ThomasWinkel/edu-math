from edu_math import Matrix

a = Matrix(1)
b = Matrix(1.1)
c = Matrix(1 + 1j)

d = Matrix([1])
e = Matrix([1,2,3])
f = Matrix([[1,2,4],[4,5,6],[7,8,9]])

print('a = ')
print(a)

print('b = ')
print(b)

print('c = ')
print(c)

print('d = ')
print(d)

print('e = ')
print(e)

print('f = ')
print(f)

print('Ones')
a = Matrix.randi(4,4,-9,9)
print(a)
print(a @ Matrix.identity(4))

q = f.inverse()

print(q)
