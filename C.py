x, k, n, p = map(float, input().split())
res = x
for i in range(int(k)):
    s, c = 0, 0
    for j in range(int(n)):
        a, b = map(int, input().split())
        s += a
        c += b
    res = (1-p)*(s*res + c)
print(res)
