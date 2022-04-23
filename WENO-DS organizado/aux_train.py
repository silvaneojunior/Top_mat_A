import tensorflow as tf
import tensorflow.keras as keras
import dill

gpus= tf.config.experimental.list_physical_devices('GPU') # Listando as placas de vídeo
tf.config.experimental.set_memory_growth(gpus[0], True)   # Selecionando a primeira GPU e configurando

def load_model(path,equation=None,WENO_method=None,conv_size=None,regul_weight=None,p=None,ativ_func=None):
    with open(path+'.cfg','rb') as file:
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

    Sim_layer = WENO(equation=config['equation'],WENO_method=config['WENO_method'],conv_size=config['conv_size'],regul_weight=regul_weight,p=p,ativ_func=ativ_func)

    # Definindo o input da rede e o otimizador de treino
    input_x   = keras.layers.Input([200], dtype='float64')
    optimizer = keras.optimizers.Adam(learning_rate=10**-3, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

    # Criando a rede neural
    Network = keras.Model(input_x, Sim_layer.Sim_step_graph(input_x))
    # Configurando a função de perda e o otimizador
    Network.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['MSE'])
    # Carregando os pesos da rede neural treinados
    #Network.load_weights('Modelo artigo')

    # Carregando os pesos treinados
    Network.load_weights(path+'.h5')
    return Network

def save_model(Network,path):
    with open(path+'.cfg','wb') as file:
        dill.dump(Network.config,file)
    Network.save_weights(path+'.h5')