import tensorflow as tf
import tensorflow.keras as keras
import dill
import os
from aux_func_V3 import *

gpus= tf.config.experimental.list_physical_devices('GPU') # Listando as placas de vídeo
if len(gpus)>0:
    tf.config.experimental.set_memory_growth(gpus[0], True)   # Selecionando a primeira GPU e configurando

def load_model(path,equation=None,WENO_method=None,conv_size=None,regul_weight=None,p=None,ativ_func=None):
    with open(path+'/config.cfg','rb') as file:
        config=dill.load(file)
    if equation is not None:
        config['equation']=equation
    if WENO_method is not None:
        config['WENO_method']=WENO_method
    if conv_size is not None:
        config['conv_size']=conv_size
    if regul_weight is not None:
        config['regul_weight']=regul_weight
    if p is not None:
        config['p']=p
    if ativ_func is not None:
        config['ativ_func']=ativ_func

    Sim_layer = WENO_layer(equation=config['equation'],WENO_method=config['WENO_method'],conv_size=config['conv_size'],regul_weight=config['regul_weight'],p=config['p'],ativ_func=config['ativ_func'])

    # Definindo o input da rede e o otimizador de treino
    input_x   = keras.layers.Input([200], dtype='float64')
    # Criando a rede neural
    Network = keras.Model(input_x, Sim_layer(input_x,0.01,0.01,FronteiraPeriodica))
    # Configurando a função de perda e o otimizador
    Network.compile()

    # Carregando os pesos treinados
    Network.load_weights(path+'/network.h5')
    return Sim_layer

def save_model(Network,path):
    if not(os.path.isdir(path+'/')):
        os.mkdir(path+'/')
    with open(path+'/config.cfg','wb') as file:
        dill.dump(Network.layers[min([i for i,layer in enumerate(Network.layers) if layer.name=='WENO_layer'])].config,file)
    Network.save_weights(path+'/network.h5')

def save_dataset(path,data_temporal,data_spatial,data_base,Δx,Δt,CFL,fronteira,equation,seed=None):
    data_dic={'data_temporal':data_temporal,
                'data_spatial': data_spatial,
                'data_base':    data_base,
                'Δx':           Δx,
                'Δt':           Δt,
                'CFL':          CFL,
                'fronteira':    fronteira,
                'equation':     equation}
    with open(path+'.bkp','wb') as file:
        dill.dump(data_dic,file)
    with open(path+'.cfg','w',encoding="utf-8") as file:
        file.writelines(f"'Δx': {Δx}\n")
        file.writelines(f"'Δt': {Δt}\n")
        file.writelines(f"'CFL': {CFL}\n")
        file.writelines(f"'fronteira': {fronteira}\n")
        file.writelines(f"'equation': {equation}\n")
        file.writelines(f"'seed': {seed}\n")

def load_dataset(path):
    with open(path+'.bkp','rb') as file:
        data_dic=dill.load(file)
    with open(path+'.cfg','r',encoding="utf-8") as file:
        config=file.read()
    print(config)
    return data_dic['data_temporal'],data_dic['data_spatial'],data_dic['data_base'],data_dic['Δx'],data_dic['Δt'],data_dic['CFL'],data_dic['fronteira'],data_dic['equation']

def create_dataset(n,poly_grade,seno_ampli,gauss_var,Δx,Δt,Total_time,granul_ref,seizures,fronteira,equation):
    y_list=[]

    # Criando funções base
    if poly_grade!=False:
        k=poly_grade+1
        pesos=np.random.uniform(size=[n,k],low=-1,high=1)
        ordem=np.floor(np.random.uniform(size=[n,1],low=1,high=k))+np.asarray([range(k)])
        pesos=(1-np.sum(tf.one_hot(ordem,k),axis=1))*pesos

        Δx_ref=Δx/granul_ref
        x_ref=np.expand_dims(np.arange(-1,1,Δx_ref),axis=0)**np.expand_dims(np.arange(k),axis=1)
        
        y_prop=np.matmul(pesos,x_ref)
        y_prop=y_prop/np.max(np.abs(y_prop),1,keepdims=True)

        y_list.append(y_prop)

    if seno_ampli!=False:
        Δx_ref=Δx/granul_ref
        x_ref=np.arange(-1,1,Δx_ref)

        k1 = np.random.uniform(1, seno_ampli, [n,1]).astype('int32')   # Amostrando uma frequência aleatória para a função seno
        k2 = np.random.uniform(1, seno_ampli, [n,1]).astype('int32')   # Amostrando uma frequência aleatória para a função seno
        a  = np.random.uniform(0, 1, [n,1]) # Amostrando um peso aleatória para ponderar as funções seno
        #b  = np.random.uniform(0, 2, [n,1]) # Amostrando um modificador de amplitude aleatório
        u1 =     a * tf.math.sin(k1*pi*x_ref) # Gerando pontos de acordo com a primeira função seno
        u2 = (1-a) * tf.math.sin(k2*pi*x_ref) # Gerando pontos de acordo com a segunda função seno
        
        y_prop=(u1+u2)
        y_prop=y_prop/np.max(np.abs(y_prop),1,keepdims=True)

        y_list.append(y_prop)

    if gauss_var!=False:
        Δx_ref=Δx/granul_ref
        x_ref=np.arange(-1,1,Δx_ref)

        k = np.random.uniform(6, 6+gauss_var, [n,1]).astype('int32')   # Amostrando uma frequência aleatória para a gaussiana
        u = np.exp(-k*(x_ref**2)) # Gerando pontos de acordo com a primeira função seno

        y_list.append(u)

    y=np.concatenate(y_list,axis=0)

    # Criando discontinuidades
    seizures+=1
    polis=np.floor(np.random.uniform(size=[y.shape[0],seizures],low=0,high=y.shape[0]))
    y0=[y[polis[:,i].astype('int32')] for i in range(seizures)]
    #position=np.floor(np.random.uniform(size=[y.shape[0]],low=0,high=y.shape[1])).astype('int32')
    position=np.round(np.linspace(1,y.shape[1]-1,seizures)).astype('int32')

    y1=y[...]
    for i,j in enumerate(position):
        y1[:,j:]=y0[i][:,j:]
    y=y1

    # Criando dados com evolução temporal e espacial

    WENO_Z_ref_sim=simulation(API_Numpy,equation,WENO_Z)
    WENO_Z_ref_derivada=lambda u0, Δx:WENO_Z_ref_sim.DerivadaEspacial(u0,Δx, fronteira)
    WENO_Z_ref_sim_step=lambda u0, Δx:WENO_Z_ref_sim.Sim_step(u0, Δt, Δx, fronteira)

    dy=[]
    for i,y_i in enumerate(np.split(y,100 if n>200 else 1,axis=0)):
        #print('Spatial: ',i,end='\r')
        dy.append(WENO_Z_ref_derivada(y_i,Δx_ref))
    dy=np.concatenate(dy,axis=0)

    y_t=[y]
    for j in range(Total_time):
        #print('Temporal: ',str(j).zfill(len(str(Total_time))),'                                 ',end='\r')
        y_t.append(WENO_Z_ref_sim_step(y_t[-1],Δx_ref))
    y_t=np.stack(y_t,axis=1)

    return y[...,np.arange(0,y.shape[-1],granul_ref)],dy[...,np.arange(0,y.shape[-1],granul_ref)],y_t[...,np.arange(0,y.shape[-1],granul_ref)]