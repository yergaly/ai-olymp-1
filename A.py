n, k = map(int,input().split())
a = list(map(int,input().split()))

b = []
for i in range(1,n):
    x = a[i]-a[i-1]
    b.append(x*x)
 
k-=1
mx = sum(b[:k])
cur = mx
#print(mx,b)

for i in range(k,len(b)):
    cur = cur+b[i]-b[i-k]
    mx = max(cur,mx)
    #print(b[i],b[i-k],mx)

print(mx)
