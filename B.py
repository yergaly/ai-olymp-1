n = int(input().strip())
a = []
for _ in range(n):
    y, p = input().split()
    a.append((int(y), float(p)))
a.sort(key=lambda x: x[1])
b = [0] * (n+1)
for i in range(n-1,-1,-1):
    b[i]=b[i+1]+a[i][0]
res = []
for i in range(n-1):
    cur = b[i]/(n-i)
    x = b[i+1]/(n-i-1)
    if cur > x:
        res.append(str(a[i][1]))

print('\n'.join(res) if res else 0)
