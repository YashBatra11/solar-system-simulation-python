import math,copy;
import numpy as np;
import matplotlib.pyplot as plt;
import sys;

#initial conditions
#https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
nb=11;
x1=1.00001423349; y1=0; x2=1.00258111495; y2=0; x3=0; y3=0; x4=5.2046195334; y4 = 0;
x5=0.38703759438; y5=0; x6=0.723272326629; y6=0; x7=1.52341740516; y7=0; x8=9.582355639772; y8 = 0; # [AU]
x9=19.20147650872; y9 = 0;  x10=30.04788757331; y10 = 0;  x11=39.48184537897; y11 = 0;
v1x=0; v1y=0.0172109402; v2x=0; v2y=0.0177884885; v3x=0; v3y=0; v4x=0; v4y=0.00756588309;
v5x=0; v5y=0.0273757907; v6x=0; v6y=0.0202141915; v7x=0; v7y=0.01391891; v8x=0; v8y=0.00560222;#[AU/day]
v9x=0; v9y = 0.00392733;  v10x=0; v10y =0.00311876;  v11x=0; v11y = 0.00271448;

m = [5.97e24,0.073e22,1.989e30,1898e24,0.330e24,4.87e24,0.642e24,568e24,86.8e24,102e24,0.0146e24]

#parameteres
n=8760; #T=n*dt
dt=0.0416667; # dt [day]
g=1.488e-34; #gravity constant [AU^3/(kg*day^2)]

S = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9],[x10,y10],[x11,y11],[v1x,v1y],[v2x,v2y],[v3x,v3y],[v4x,v4y],[v5x,v5y],[v6x,v6y],[v7x,v7y],[v8x,v8y],[v9x,v9y],[v10x,v10y],[v11x,v11y]] #matrix of current state
Sn=copy.deepcopy(S); #next state

def ras(r1, r2): #pythagoras
    return math.sqrt((r1*r1) + (r2*r2));

def sila(B,t): #sum of all forces on planet
    A=copy.deepcopy(B);
    Fx=0;
    Fy=0;
    for i in range(nb):
        if(i!=t):
            Fx=Fx-g*m[i]*m[t]*(A[t][0]-A[i][0])/ras(A[t][0]-A[i][0],A[t][1]-A[i][1])**3;
            Fy=Fy-g*m[i]*m[t]*(A[t][1]-A[i][1])/ras(A[t][0]-A[i][0],A[t][1]-A[i][1])**3;
    return [Fx,Fy];

def dif(S): #step matrix [V,a];  dif([r,V]) => [V,a];
    B = copy.deepcopy(S);
    out = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]];
    for i in range(nb):
        out[i+nb] = [sila(B,i)[0]/m[i],sila(B,i)[1]/m[i]];
    for i in range(nb):
        out[i]=[B[i+nb][0],B[i+nb][1]];
    return out;

def zbir(A,B): #sum of two matrices
    C= copy.deepcopy(A);
    for i in range(nb*2):
        for j in range(2):
            C[i][j]=C[i][j]+B[i][j];
    return C;

def proizvod(M,s): #multiplication of matrix and scalar
    N = copy.deepcopy(M);
    for i in range(nb*2):
        for j in range(2):
            N[i][j]=N[i][j]*s;
    return N;

def energija(A): #system energy
    B=copy.deepcopy(A);
    U=0; T=0; E=0;
    U=U-g*m[0]*m[1]/ras(S[0][0]-S[1][0],S[0][1]-S[1][1])-g*m[0]*m[2]/ras(S[0][0]-S[2][0],S[0][1]-S[2][1])-g*m[2]*m[1]/ras(S[2][0]-S[1][0],S[2][1]-S[1][1]);
    for i in range(nb):
        for j in range(2):
            T=T+0.5*m[i]*S[i+1][j]**2
    E=T+U;
    return E;

#cache for drawing
x, y = [[], [], [], [],[],[],[],[],[],[],[]], [[], [], [], [],[],[],[],[],[],[],[]];
t= np.zeros(n); E = np.zeros(n);
size = [80,1,1500,880,24,75,85,760,320,310,1]
clr = ['g','grey','gold','sienna','silver','indianred','red','coral','blue','royalblue','tan'];
#progress
leng = 20;

#simulation
for i in range(n):
    #E[i]=energija(copy.deepcopy(S));
    #cache
    t[i]=i*dt;
    for j in range(nb):
        x[j].append(S[j][0]);
        y[j].append(S[j][1]);
    #rk4
    k1=dif(copy.deepcopy(S));
    k2=dif(zbir(copy.deepcopy(S),proizvod(copy.deepcopy(k1),dt/2)));
    k3=dif(zbir(copy.deepcopy(S),proizvod(copy.deepcopy(k2),dt/2)));
    k4=dif(zbir(copy.deepcopy(S),proizvod(copy.deepcopy(k3),dt)));
    k1k2=zbir(copy.deepcopy(k1),proizvod(copy.deepcopy(k2),2));
    k3k4=zbir(proizvod(copy.deepcopy(k3),2),copy.deepcopy(k4));
    rk=proizvod(zbir(copy.deepcopy(k1k2),copy.deepcopy(k3k4)),dt/6);
    Sn=zbir(copy.deepcopy(Sn),copy.deepcopy(rk));
    #set new state
    S=copy.deepcopy(Sn);
    #progress bar
    block = int(round(leng*i/n))
    msg = "\r{0}: [{1}] {2}%".format("Simulacija", "#"*block + "-"*(leng-block), round(i/n*100, 2))
    sys.stdout.write(msg); sys.stdout.flush();

msg = "\r{0}: [{1}] {2}% DONE\r\n".format("Simulacija", "#"*block + "-"*(leng-block), round(n/n*100, 2))
sys.stdout.write(msg); sys.stdout.flush();

#plt.plot(E,t);
plt.title("T={} D \n dt ={} D".format(n*dt,dt)); plt.xlabel('x [AU]'); plt.ylabel('y [AU]');
for i in range(nb):
    plt.scatter(x[i][0],y[i][0],color=clr[i],s=size[i]);
    plt.plot(x[i], y[i], color=clr[i]);
print(S[1])
plt.show();
