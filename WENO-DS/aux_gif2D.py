import API_Numpy
import dill
# from IPython.display import clear_output
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gc

def helper_plot(ref_U,x,y,name,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None,colorbar=True):
    plt.figure(figsize=figsize)
    a=plt.pcolormesh(x,y,ref_U,cmap='jet',vmin=vmin,vmax=vmax)
    if colorbar:
        plt.colorbar(a)
    plt.contour(x,y,ref_U,vmin=vmin,vmax=vmax,levels=levels,colors='black',linewidths=0.25)
    plt.title(name)
    plt.xlim(xlim)
    plt.ylim(ylim)

def create_movie(label,N,name,x,y,figsize,vmin=None,vmax=None,levels=None,xlim=None,ylim=None,colorbar=True,save_dir=None,start_time=0,end_time=-1):
    if save_dir is None:
        save_dir=label
    if not(os.path.isdir(f'imagens/{save_dir}-{N}-{name}/')):
        os.mkdir(f'imagens/{save_dir}-{N}-{name}/')

    
    with open(f'imagens/{label}-{N}-{name}/cache.bkp','rb') as file:
        time_total,count_total,time_acum=dill.load(file)
    if(end_time<0 or end_time>count_total):
        end_time=count_total
        
    frames = []
    for count in range(start_time+1,end_time+1):

        with open(f'imagens/{label}-{N}-{name}/backup_{count}.bkp','rb') as file:
            U=dill.load(file)[0]
        helper_plot(U,x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar)
        plt.savefig(f'imagens/{save_dir}-{N}-{name}/{str(count).zfill(4)}.png')
        
        plt.clf()
        plt.close()
        gc.collect()
        # frames.append(helper_gif(U,x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar))

        new_frame = Image.open(f'imagens/{save_dir}-{N}-{name}/{str(count).zfill(4)}.png')
        frames.append(new_frame)

        print(count,'                             ',end='\r')

    # Save into a GIF file that loops forever
    frames[0].save(f"imagens/{save_dir}-{N}-{name}/{name}.gif",
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=1, loop=0)
    helper_plot(U,x,y,name,figsize,vmin=vmin,vmax=vmax,levels=levels,xlim=xlim,ylim=ylim,colorbar=colorbar)
    plt.savefig(f'imagens/{save_dir}-{N}-{name}/{name}.png')
    plt.clf()
    plt.close()
    gc.collect()
    # gif.save(frames,
    #         f"imagens/{save_dir}-{N}-{name}/{name}.gif", 
    #         duration=1)
    # Create the frames

def create_data(label,N,name,U0,WENO,Δt_max,t_final,cfl,Δx, Δy, GhostPointsX, GhostPointsY,Force,continue_flag=True):
    if not(os.path.isdir(f'imagens/{label}-{N}-{name}/')):
        os.mkdir(f'imagens/{label}-{N}-{name}/')
    if os.path.isfile(f'imagens/{label}-{N}-{name}/cache.bkp') and continue_flag:
        with open(f'imagens/{label}-{N}-{name}/cache.bkp','rb') as file:
            cur_time,cur_count,cur_acum=dill.load(file)
        with open(f'imagens/{label}-{N}-{name}/backup_{cur_count}.bkp','rb') as file:
            cur_val=dill.load(file)
        U=cur_val
    else:
        U=U0.copy()
        cur_count=0
        cur_time=0
        cur_acum=0
    t=cur_time
    count=cur_count
    spent_acum=cur_acum
    print(name)
    print(f'Tempo inicial: {t}')
    time_init=time.time()
    while t<t_final:
        
        t_step=0
        while t_step<Δt_max:
            
            Λ  = WENO.equation.maximum_speed(U)
            Δt = min(Δx,Δy)*cfl/Λ
            Δt = min(Δt_max,Δt)
            Δt = np.where(t_step+Δt>Δt_max,Δt_max-t_step,Δt)

            U=WENO.Sim_step(U, Δt, Δx, Δy, GhostPointsX, GhostPointsY,Force, t=t+t_step+Δt).numpy()
            t_step+=Δt
        t+=Δt_max
        count+=1
        time_spent=time.time()-time_init+spent_acum

        with open(f'imagens/{label}-{N}-{name}/cache.bkp','wb') as file:
            dill.dump([t,count,time_spent],file)
        with open(f'imagens/{label}-{N}-{name}/backup_{count}.bkp','wb') as file:
            dill.dump(U,file)
        perc_done=t/t_final
        perc_not_done=1-perc_done
        expe_tot_time=time_spent/perc_done
        ETA=perc_not_done*expe_tot_time
        print(f'Tempo atual: {t} - ETA:{got_the_time(ETA)}                     ',end='\r')
    print(f'Tempo final: {t}')

def create_data_last_frame(label,N,name,U0,WENO,t_final,cfl,Δx, Δy, GhostPointsX, GhostPointsY,Force,continue_flag=True):
    if not(os.path.isdir(f'imagens/{label}-{N}-{name}/')):
        os.mkdir(f'imagens/{label}-{N}-{name}/')
    if os.path.isfile(f'imagens/{label}-{N}-{name}/cache.bkp') and continue_flag:
        with open(f'imagens/{label}-{N}-{name}/cache.bkp','rb') as file:
            cur_time,cur_count,cur_acum=dill.load(file)
        with open(f'imagens/{label}-{N}-{name}/backup_{cur_count}.bkp','rb') as file:
            cur_val=dill.load(file)
        U=cur_val.copy()
    else:
        U=U0.copy()
        cur_val=0
        cur_time=0
    t=cur_time
    count=cur_count
    print(name)
    time_acum=cur_acum
    time_init=time.time()
    print(f'Tempo inicial: {t}')
    while t<t_final:
        Λ  = WENO.equation.maximum_speed(U)
        Δt = min(Δx,Δy)*cfl/Λ
        Δt = np.where(t+Δt>t_final,t_final-t,Δt)

        U=WENO.Sim_step(U, Δt, Δx, Δy, GhostPointsX, GhostPointsY,Force, t=t+Δt).numpy()
        t+=Δt
        count+=1
        time_cur=time.time()
        time_spent=time_cur-time_init+time_acum
        
        with open(f'imagens/{label}-{N}-{name}/cache.bkp','wb') as file:
            dill.dump([t,count,time_spent],file)
        with open(f'imagens/{label}-{N}-{name}/backup_{count}.bkp','wb') as file:
            dill.dump(U,file)
        
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