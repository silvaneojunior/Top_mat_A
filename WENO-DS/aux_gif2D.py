import gif
import API_Numpy
import dill
import os
import matplotlib.pyplot as plt
import numpy as np

def helper_plot(ref_U,x,y,name,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None):
    plt.figure(figsize=figsize)
    a=plt.pcolormesh(x,y,ref_U,cmap='jet',vmin=vmin,vmax=vmax)
    plt.colorbar(a)
    plt.contour(x,y,ref_U,vmin=vmin,vmax=vmax,levels=levels,colors='black',linewidths=0.25)
    plt.title(name)
    plt.xlim(xlim)
    plt.ylim(ylim)

helper_gif=gif.frame(helper_plot)

def create_movie(label,N,name,x,y,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None):
    frames=[]

    with open(f'imagens/{label}-{N}-{name}/data.bkp','rb') as file:
        U_total=dill.load(file)
    for count,U in enumerate(API_Numpy.unstack(U_total,axis=0)):

        helper_plot(U[0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim)
        plt.savefig(f'imagens/{label}-{N}-{name}/{str(count).zfill(4)}.png')
        plt.close()

        frames.append(helper_gif(U[0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim))

        print(count,'                             ',end='\r')
    helper_plot(U_total[-1][0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim)
    plt.savefig(f'imagens/{label}-{N}-{name}/{name}.png')
    plt.close()
    gif.save(frames,
            f"imagens/{label}-{N}-{name}/{name}.gif", 
            duration=1)

def create_data(label,N,name,U0,WENO,Δt_max,t_final,cfl,Δx, Δy, GhostPointsX, GhostPointsY,Force,continue_flag=True):
    if not(os.path.isdir(f'imagens/{label}-{N}-{name}/')):
        os.mkdir(f'imagens/{label}-{N}-{name}/')
    if os.path.isfile(f'imagens/{label}-{N}-{name}/data.bkp') and continue_flag:
        with open(f'imagens/{label}-{N}-{name}/data.bkp','rb') as file:
            U_total=list(API_Numpy.unstack(dill.load(file),axis=0))
        U=U_total[-1].copy()
    else:
        U=U0.copy()
        U_total=[U0.copy()]
    count=len(U_total)-1
    t=count*Δt_max
    print(name)
    print(f'Tempo inicial: {t}')
    while t<t_final:
        
        t_step=0
        while t_step<Δt_max:
            
            Λ  = WENO.equation.maximum_speed(U)
            Δt = min(Δx,Δy)*cfl/Λ
            Δt = min(Δt_max,Δt)
            Δt = np.where(t_step+Δt>Δt_max,Δt_max-t_step,Δt)

            U=WENO.Sim_step(U, Δt, Δx, Δy, GhostPointsX, GhostPointsY,Force, t=t+t_step+Δt)
            t_step+=Δt
        t+=Δt_max
        count+=1
        U_total.append(U)

        with open(f'imagens/{label}-{N}-{name}/data.bkp','wb') as file:
            dill.dump(np.stack(U_total,axis=0),file)

        print(f'Tempo atual: {t}                     ',end='\r')
    print(f'Tempo final: {t}')