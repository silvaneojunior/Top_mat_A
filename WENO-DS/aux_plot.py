from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import imageio as img
import time
import dill
import os 

f_test_1 = lambda x: -np.sin(np.pi*x) - 0.5 * x**3 + np.where(x < 0, 0.0, 1.0)
f_test_2 = lambda x: np.exp(-10*(x**2))

def f_test_3(x):
    z = tf.constant(-0.7, dtype=float_pres)
    δ = tf.constant(0.005, dtype=float_pres)
    β = tf.math.log(tf.constant(2.0, dtype=float_pres))/(36.0*(δ**2.0))
    a = tf.constant(0.5, dtype=float_pres)
    α = tf.constant(10, dtype=float_pres)
    
    def G(x, β, z):
        return tf.math.exp(-β*(x-z)**2)
    
    def F(x, α, a):
        return tf.math.sqrt(tf.math.maximum(1-(α**2)*((x-a)**2),0))
    
    f1 = (G(x, β, z-δ) + 4*G(x, β, z) + G(x, β, z+δ))/6 # x in [-0.8, -0.6]
    f2 = 1                                              # x in [-0.4, -0.2]
    f3 = 1-tf.math.abs(10*(x-0.1))                      # x in [ 0.0,  0.2]
    f4 = (F(x, α, a-δ) + 4*F(x, α, a) + F(x, α, a+δ))/6 # x in [ 0.4,  0.6]
    f5 = 0                                              # otherwise
    
    condition = tf.math.logical_and(
        tf.math.greater_equal(x, -0.8),
        tf.math.less_equal(x, -0.6)
    )
    f1 = f1*tf.cast(tf.where(condition, 1.0, 0.0), dtype=float_pres)
    
    condition = tf.math.logical_and(
        tf.math.greater_equal(x, -0.4),
        tf.math.less_equal(x, -0.2)
    )
    f2 = f2*tf.cast(tf.where(condition, 1.0, 0.0), dtype=float_pres)
    
    condition = tf.math.logical_and(
        tf.math.greater_equal(x, 0.0),
        tf.math.less_equal(x, 0.2)
    )
    f3 = f3*tf.cast(tf.where(condition, 1.0, 0.0), dtype=float_pres)
    
    condition = tf.math.logical_and(
        tf.math.greater_equal(x, 0.4),
        tf.math.less_equal(x, 0.6)
    )
    f4 = f4*tf.cast(tf.where(condition, 1.0, 0.0), dtype=float_pres)
    
    f = f1 + f2 + f3 + f4 + f5
    
    return f

df_test_1 = lambda x: -np.pi*np.cos(np.pi*x) - 1.5 * x**2
df_test_2 = lambda x: -20*x*np.exp(-10*(x**2))

def create_f_points(f_test, Δx, xlim=(-1,1), dtype='float64'):
    
    x = np.arange(xlim[0], xlim[-1], Δx, dtype=dtype) # Gerando a malha de pontos no espaço unidimensional
    u = f_test(x)                                     # Obtendo a condição inicial a partir de f_test
    u = np.expand_dims(u, axis=0)                     # Acrescentando uma dimensão
    
    x = np.tile(x, 4)
    
    return x, u

def get_inner_val(test_weno, u, Δx, fronteira):
    
    n_pontos = u.shape[1]
    
    ω_plot = np.zeros([len(test_weno),n_pontos,3])
    α_plot = np.zeros([len(test_weno),n_pontos,3])
    β_plot = np.zeros([len(test_weno),n_pontos,3])
    δ_plot = np.zeros([len(test_weno),n_pontos,3])

    for i, weno in enumerate(test_weno):
        
        ω, α, β, δ = weno.Get_weights(u, Δx, fronteira)

        ω_plot[i] = np.squeeze(ω)
        α_plot[i] = np.squeeze(α)
        β_plot[i] = np.squeeze(β)
        δ_plot[i] = np.squeeze(δ)
        
    return ω_plot, α_plot, β_plot, δ_plot

def create_compara_plot(x, values, labels, is_log=True, xlim=(-1,1), ylim=(10**-8,(10**-0)), hlines=True):

    fig, axs = plt.subplots(1, len(values), figsize=(8*len(values), 6))

    for i, y_plot, name in zip(range(len(values)), values, labels):
        if len(values)>1:
            ax = axs[i]
        else:
            ax = axs

        # Inserindo os pontos nos gráficos
        ax.plot(x, y_plot[:,0], 's', color='red'  , alpha=1, label='Esquerda')
        ax.plot(x, y_plot[:,1], 'D', color='black', alpha=1, label='Central')
        ax.plot(x, y_plot[:,2], 'o', color='blue' , alpha=1, label='Direita')

        if hlines:
            # Criando as linhas horizontais dos gráficos
            ax.hlines(
                y          = 1/10,
                xmin       = x[0],
                xmax       = x[-1],
                color      = 'red',
                linestyles = '--'
            )
            ax.hlines(
                y          = 6/10,
                xmin       = x[0],
                xmax       = x[-1],
                color      = 'black',
                linestyles = '--'
            )
            ax.hlines(
                y          = 3/10,
                xmin       = x[0],
                xmax       = x[-1],
                color      = 'blue',
                linestyles = '--'
            )

        # Configuração de parâmetros gráficos referentes ao eixo y
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if is_log:
            ax.set_yscale('log')
            ax.yaxis.set_ticks(10**np.arange(np.log10(ylim[0]), np.log10(ylim[1]), 1.0))
        ax.grid(ls='--')
        ax.title.set_text(name)
        ax.legend()

class compara_evolve:
    def __init__(self,WENOs,Δx,malha,names,f_test,fronteira,f_transform=lambda x:x,CFL=0.5,x_range=(-1,1),xlim=(-1,1),ylim=(-0.1, 1.1),use_cache=None,replace=False,colors=['black','#ffaa55', '#55aaff'],shapes=['--','d','o'],figsize=(16,8)):

        self.WENOs=WENOs
        self.malha=malha
        self.lines=[]
        self.U=[]
        self.ax=[]
        self.Δx=Δx
        self.CFL=CFL
        self.fronteira=fronteira
        self.names=names
        self.use_cache=use_cache if use_cache is not None or not(self.replace) else [False]*len(names)
        self.replace=replace
        self.f_transform=f_transform

        self.fig = plt.figure(1, constrained_layout=True, figsize=figsize)
        self.ax=self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        for j, name, color, shape in zip(malha, names, colors, shapes):
            
            x, U_i = create_f_points(f_test=f_test, Δx=Δx/j, xlim=x_range)
            self.U.append(U_i)
            self.ax.set_ylim(*ylim)
            self.lines.append(self.ax.plot(x, f_transform(np.squeeze(U_i)),shape,label=name,color=color))
        self.ax.legend()
        self.hfig = display(self.fig, display_id=True)
        
    def update(self, Δt):
        
        for i, line, Sim, k, use_cache in zip(range(len( self.names)), self.lines, self.WENOs, self.malha, self.use_cache):
            if use_cache:
                self.U[i] = self.cache[i][:,len(self.history)]
            else:
                self.U[i] = Sim(self.U[i],
                        np.float64(Δt),
                        np.float64(self.Δx/k),
                        np.float64(self.CFL),
                        self.fronteira
                            )
            squeezed_u=self.f_transform(np.squeeze(self.U[i]))
                
            # Exibindo graficamente os valores obtidos
            line[0].set_ydata(squeezed_u)
        
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def gif_framework(self, frames, Δt, path):

        if not(os.path.isdir(path+'/')):
            os.mkdir(path+'/')
        if not(os.path.isdir(path+'/frames/')):
            os.mkdir(path+'/frames/')

        
        for index, flag_cache, name in zip(range(len(self.names)), self.use_cache, self.names):
            if flag_cache:
                if not(os.path.isfile(path+'/{}.mat'.format(name))):
                    self.use_cache[index]=False

        self.cache = []
        for flag_cache, name in zip(self.use_cache, self.names):
            if flag_cache:
                with open(path+'/{}.mat'.format(name),'rb') as file:
                    self.cache.append(dill.load(file))
            else:
                self.cache.append(None)

        self.history = []
        with img.get_writer(path+'/plot.gif', mode='I') as writer:
            for i in range(frames):
                self.update(Δt)
                self.history.append(self.U[:])
                plt.savefig(path+'/frames/{}.png'.format(str(i).zfill(len(str(frames)))))
                image = img.imread(path+'/frames/{}.png'.format(str(i).zfill(len(str(frames)))))
                writer.append_data(image)
        for i,name in enumerate(self.names):
            data_mat = [Ui[i] for Ui in self.history]
            data_mat = np.stack(data_mat, 1)
            with open(path+'/{}.mat'.format(name),'wb') as file:
                dill.dump(data_mat, file)