import numpy as np 
import matplotlib.pyplot as plt # 図の作成環境のロード
from matplotlib import cm # カラーマップ
from tqdm import tqdm # プログレスバーを表示
import time # 計算時間計測プロファイリング用

from IPython.display import clear_output # 途中結果の図示用
from numba import double
from numba import jit 
# parameters 
# computational domain
Lx=0.1e0
Ly=Lx

# wall velocity
Uwall=0.01
# fluid property
nu=1.e-6 # kinematic viscosity (=mu/rho)
# rho=1.e3 # density

# nondimensional time (in L/Uwall)
endT=20

# mesh girds
Nx=41
Ny=43

# for dt
CFL=0.5
CFLv=0.8

# for solver
accel = 1.925e0
err_tol = 1.e-6
tiny = 1.e-20
# set grid 
dx=Lx/np.float64(Nx-1)
dy=Ly/np.float64(Ny-1)

# mesh information (grid)
x=np.array(np.zeros(Nx+2),dtype=np.float64)
y=np.array(np.zeros(Ny+2),dtype=np.float64)
# cell centre position
xc=np.array(np.zeros(Nx+1),dtype=np.float64)
yc=np.array(np.zeros(Ny+1),dtype=np.float64)

# set uniform mesh
x[0]=-dx
for i in range(1,Nx+2):
  x[i]=x[i-1]+dx; # raw grid
  xc[i-1]=0.5*(x[i-1]+x[i])/Lx # scaled axis of cell centre

y[0]=-dy
for i in range(1,Ny+2):
  y[i]=y[i-1]+dy; # raw grid 
  yc[i-1]=0.5*(y[i-1]+y[i])/Lx # scaled axis of cell centre

x2d, y2d = np.meshgrid(x,y) # for vector plots
if Uwall != 0.0:
  Uref = Uwall
else:
  Uref = nu/Lx
    
dt = min(CFL*dx/Uref, CFLv*dx*dx/nu)
Nt = int(endT*Lx/Uref/dt)

dx2=(dx*dx) 
dy2=(dy*dy)
# variables 
u=np.zeros([Ny+1, Nx+2],dtype=np.float64)
v=np.zeros([Ny+2, Nx+1],dtype=np.float64)
p=np.zeros([Ny+1, Nx+1],dtype=np.float64)

uaux=np.zeros([Ny+1, Nx+2],dtype=np.float64)
vaux=np.zeros([Ny+2, Nx+1],dtype=np.float64)
dive=np.zeros([Ny+1, Nx+1],dtype=np.float64)
    
# for plot the velocity on regular grid
ur=np.array(np.zeros((Ny, Nx+2),dtype=np.float64))
vr=np.array(np.zeros((Ny+2, Nx),dtype=np.float64))
@jit
def calc_aux_u(uaux,u,v):
    for jc in range(1, Ny):
        for i in range(1, Nx+1):
            visc = (u[jc, i-1]-2e0*u[jc,i]+u[jc, i+1])/dx2 \
                    +(u[jc-1, i]-2e0*u[jc,i]+u[jc+1, i])/dy2
            conv = (+( ( u[jc, i-1] + u[jc, i])/2e0 \
                              *(-u[jc, i-1]+u[jc, i])/dx ) \
                          +( ( u[jc, i]+u[jc, i+1])/2e0 \
                              *(-u[jc, i]+u[jc, i+1])/dx ) \
                         )/2e0 \
                        +(+( ( v[jc, i-1]+v[jc, i])/2e0 \
                               *(-u[jc-1,i]+u[jc, i])/dy ) \
                            +( ( v[jc+1, i-1] + v[jc+1,i])/2e0 \
                               *(-u[jc,i]+u[jc+1,i])/dy ) \
                         )/2e0
            uaux[jc,i] = u[jc,i] + dt*(-conv + nu*visc)
    
def set_bc_u(u):
    # left and right walls 
    for jc in range(0,Ny+1): 
        u[jc,1] =0.e0; 
        u[jc,Nx]=0.e0
        #u[jc,0] = -u[jc,2] # left imaginary cell (for visualization)
        #u[jc,Nx+1] = -u[jc,Nx-1] # right imaginary cell (for visualization)

    # bottom and top walls (embedded)
    for i in range(0,Nx+2):
        u[0,i] = -u[1,i]  # bottom wall (uc=0)
        u[Ny,i] = -u[Ny-1,i]+2.e0*Uwall # moving wall (uc=Uwall)
      @jit
def calc_aux_v(vaux,u,v):
    for j in range(1, Ny+1):
        for ic in range(1, Nx):
            # 
            visc = (v[j-1, ic]-2e0*v[j, ic]+v[j+1, ic])/dy2 \
                    +(v[j, ic-1]-2e0*v[j, ic]+v[j, ic+1])/dx2
            conv = (+( ( u[j-1, ic]+u[j, ic])/2e0 \
                              *(-v[j, ic-1]+v[j, ic])/dx ) \
                          +( ( u[j-1, ic+1]+u[j, ic+1])/2e0 \
                              *(-v[j, ic]+v[j, ic+1])/dx ) \
                         )/2e0 \
                       +(+( ( v[j-1, ic]+v[j, ic])/2e0 \
                              *(-v[j-1, ic]+v[j, ic])/dy ) \
                           +( ( v[j, ic]+v[j+1, ic])/2e0 \
                              *(-v[j, ic]+v[j+1, ic])/dy ) \
                         )/2e0
            vaux[j, ic] = v[j, ic] + dt*(-conv + nu*visc) 

def set_bc_v(v):
    # left and right walls (embedded)
    for j in range(0,Ny+2):    
        v[j,0] = -v[j,1]
        v[j,Nx]= -v[j,Nx-1]

    # bottom and top walls (on walls)
    for ic in range(0,Nx+1):
        v[1,ic]  =0.e0
        v[Ny,ic] =0.e0
        #v[0,ic]  = -v[2,ic]
        #v[Ny+1,ic] = -v[Ny-1,ic]
@jit
def divergence(div,u,v):
    for jc in range(1,Ny):
        for ic in range(1,Nx):
            div[jc,ic] = ( (-u[jc,ic] + u[jc, ic+1])/dx \
                       +(-v[jc,ic] + v[jc+1, ic])/dy \
                      )/dt              
          @jit
def calcP(p,div):
    err_n=0.0
    err_d=0.0
    for jc in range(1,Ny):
        for ic in range(1,Nx):
            d_pres = (  dy2*(p[jc, ic-1] + p[jc, ic+1]) \
                             + dx2*(p[jc-1,ic] + p[jc+1,ic]) \
                           - (dx2*dy2 * div[jc,ic]) )/((dx2+dy2)*2e0) - p[jc,ic]
            p[jc,ic] = p[jc,ic] + accel*d_pres
            err_n = err_n + d_pres*d_pres
            err_d = err_d + p[jc,ic]*p[jc,ic]
    set_bc_pressure(p)
    if err_d < tiny:
        err_d = 1e0
    err_r = np.sqrt(err_n/err_d)
    return err_r

@jit
def set_bc_pressure(p):
    # p[1,1]=0.e0
    for ic in range(1,Nx):
        p[0,ic] = p[1,ic]
        p[Ny,ic]= p[Ny-1,ic]
    for jc in range(1,Ny):
        p[jc,0]=p[jc,1]
        p[jc,Nx]=p[jc,Nx-1]
      @jit
def correct_u(u, uaux, p):
    for jc in range(1, Ny):
        for i in range(1, Nx+1):
            u[jc, i] = uaux[jc, i] - dt*(-p[jc, i-1] + p[jc, i])/dx

def correct_v(v, vaux, p):
    for j in range(1, Ny+1):
        for ic in range(1, Nx):
            v[j, ic] = vaux[j, ic] - dt*(-p[j-1, ic] + p[j, ic])/dy
          time_ini=time.time()
ifield=0;
for itr in tqdm(range(0,Nt)):
    t0=time.time()
    calc_aux_u(uaux, u, v)
    set_bc_u(uaux)
    calc_aux_v(vaux, u, v)
    set_bc_v(vaux)
    divergence(dive, uaux, vaux)

    err_r=1.e0; itr_SOR=0
    while err_r > err_tol:
        itr_SOR += 1
        err_r=calcP(p, dive)
        if itr < 10:
            if itr_SOR >1000:
                break
        elif itr < 20:
            if itr_SOR >5000:
                break
        elif itr < 30:
            if itr_SOR >10000:
                break
    if np.isnan(err_r)==1:
        print('NaN: at itr='+str(itr)+', itr(SOR)='+str(itr_SOR))
        break

    correct_u(u, uaux, p)
    set_bc_u(u)
    correct_v(v, vaux, p)
    set_bc_v(v)

    if np.mod(itr,100)==0:
        clear_output(True)
        fig, ax = plt.subplots()
        tcf = ax.contourf(xc, yc, p)
        #tcf = ax.contourf(xc, yc, dive) # 連続の式を満足しているかチェック
        fig.colorbar(tcf)
        # rough interpolation of velocities
        uc=0.5*(u[:,:-1]+u[:,1:])/Uref # interpolate at the regular grid with scaling
        vc=0.5*(v[:-1,:]+v[1:,:])/Uref # interpolate at the regular grid with scaling
        ax.streamplot(xc,yc,uc,vc,color='w',density=1,integration_direction='backward',arrowstyle="->")
        ax.set_aspect('equal')
        ax.set_title("$Re$={0:.2f}".format(Uwall*Ly/nu)+", $t$={0:.3f}".format(itr*dt),fontsize=20)
        plt.xlim(0, 1); plt.ylim(0, 1);
        plt.show()

t1=time.time()
print(' nstep = '+str(itr) + ': time elapsed = '+str(t1-time_ini)+' sec.')
