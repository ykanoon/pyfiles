import math

L=5
d = 0.5
pi = math.pi
lam = 1.875
rho = 2.8*10**3
E = 70*10**9
I = 0.785*10**-8

f=(lam**2)*d*math.sqrt(E/rho)/(8*pi*L**2)

print('f='+str(f))
