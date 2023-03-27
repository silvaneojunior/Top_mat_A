import gif
import API_Numpy
import dill
# from IPython.display import clear_output
import time
import os
import matplotlib.pyplot as plt
import numpy as np

def helper_plot(ref_U,x,y,name,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None,colorbar=True):
    plt.figure(figsize=figsize)
    a=plt.pcolormesh(x,y,ref_U,cmap='jet',vmin=vmin,vmax=vmax)
    if colorbar:
        plt.colorbar(a)
    plt.contour(x,y,ref_U,vmin=vmin,vmax=vmax,levels=levels,colors='black',linewidths=0.25)
    plt.title(name)
    plt.xlim(xlim)
    plt.ylim(ylim)

helper_gif=gif.frame(helper_plot)

def create_movie(label,N,name,x,y,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None,colorbar=True,save_dir=None,start_time=0,end_time=-1):
    frames=[]
    if save_dir is None:
        save_dir=label
    if not(os.path.isdir(f'imagens/{save_dir}-{N}-{name}/')):
        os.mkdir(f'imagens/{save_dir}-{N}-{name}/')

    with open(f'imagens/{label}-{N}-{name}/data.bkp','rb') as file:
        U_total=dill.load(file)
        if end_time==-1:
            end_time=U_total.shape[0]-1
        U_total=U_total[start_time:end_time+1]
    for count,U in enumerate(API_Numpy.unstack(U_total,axis=0)):

        helper_plot(U[0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar)
        plt.savefig(f'imagens/{save_dir}-{N}-{name}/{str(count).zfill(4)}.png')
        plt.close()

        frames.append(helper_gif(U[0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar))

        print(count,'                             ',end='\r')
    helper_plot(U_total[-1][0],x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar)
    plt.savefig(f'imagens/{save_dir}-{N}-{name}/{name}.png')
    plt.close()
    gif.save(frames,
            f"imagens/{save_dir}-{N}-{name}/{name}.gif", 
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

def create_data_last_frame(label,N,name,U0,WENO,t_final,cfl,Δx, Δy, GhostPointsX, GhostPointsY,Force,continue_flag=True):
    cur_val=[U0,0,0,0]
    if not(os.path.isdir(f'imagens/{label}-{N}-{name}/')):
        os.mkdir(f'imagens/{label}-{N}-{name}/')
    if os.path.isfile(f'imagens/{label}-{N}-{name}/data.bkp') and continue_flag:
        with open(f'imagens/{label}-{N}-{name}/data.bkp','rb') as file:
            cur_val=dill.load(file)
        U0=cur_val[0]
    U=U0.copy()
    t=cur_val[1]
    count=cur_val[2]
    print(name)
    time_acum=cur_val[3]
    time_init=time.time()
    print(f'Tempo inicial: {t}')
    while t<t_final:
        Λ  = WENO.equation.maximum_speed(U)
        Δt = min(Δx,Δy)*cfl/Λ
        Δt = np.where(t+Δt>t_final,t_final-t,Δt)

        U=WENO.Sim_step(U, Δt, Δx, Δy, GhostPointsX, GhostPointsY,Force, t=t+Δt)
        t+=Δt
        count+=1
        time_cur=time.time()
        time_spent=time_cur-time_init+time_acum
        
        with open(f'imagens/{label}-{N}-{name}/data.bkp','wb') as file:
            dill.dump([U.numpy(),t,count,time_spent],file)
        
        pretty_time=got_the_time(time_spent*(t_final-t)/t)
        pretty_time_total=got_the_time(time_spent*t_final/t)
    
        print(f'Tempo atual: {t.flatten()[0]} - {np.round(100*t/t_final,2).flatten()[0]}% concluido - ETA: {pretty_time} - Total time: {pretty_time_total}                           ',end='\r')
    print(f'Tempo final: {t}')
    return(U)


def got_the_time(obs_time):
    milisec,obs_time=(obs_time%1)*1000,obs_time//1
    sec,obs_time=int(obs_time%60),obs_time//60
    minutes,obs_time=int(obs_time%60),obs_time//60
    hours,obs_time=int(obs_time%24),int(obs_time//24)
    
    if obs_time>0:
        obs_time='{0} dias, {1}h, {2}min'.format(str(obs_time),str(hours),str(minutes))
    elif hours>0:
        obs_time='{0}h, {1}min'.format(str(hours),str(minutes))
    elif minutes>0:
        obs_time='{0}min, {1}s'.format(str(minutes),str(sec))
    elif sec>0:
        obs_time='{0}s e ~{1}ms'.format(str(sec),str(int(milisec)))
    else:
        obs_time='~{0}ms'.format(str(int(milisec)))
    return obs_time