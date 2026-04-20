#Import das bibliotecas que serão usadas. 
import matplotlib.pyplot as plt #matplotlib para gráficos
from matplotlib.animation import FuncAnimation #FuncAnimation para fazer a animação das trajetorias
from IPython.display import HTML #IPython display para conseguir rodar a animação mesmo em jupyter notebook
from numba import njit, prange #numba para acelerar e paralelizar os códigos
import numpy as np #numpy para manipulação numérica e de arrays

#Funcao de calcula a derivada dupla do potencial
#os inputs sao a posicao dada pelo par (x, y), a massa reduzida mu e o indice idx que se mapeia como
#idx = 0 => U_{xx} || idx = 1 => U_{xy} || idx = 2 => U_{yy}
@njit
def potential_der(x, y, mu, idx):
    termo1 = (x + mu)**2 + y**2
    termo2 = (x - 1 + mu)**2 + y**2
    
    termo1_32 = termo1**(-1.5)
    termo1_52 = termo1**(-2.5)
    
    termo2_32 = termo2**(-1.5)
    termo2_52 = termo2**(-2.5)
    if idx == 0:
        Uxx = 1 - (1 - mu) * (termo1_32 - 3 * (x + mu)**2 * termo1_52) \
            - mu * (termo2_32 - 3 * (x - 1 + mu)**2 * termo2_52)
        return Uxx
    elif idx == 1:
        Uxy = 3 * (1 - mu) * (x + mu) * y * termo1_52 \
        + 3 * mu * (x - 1 + mu) * y * termo2_52
        return Uxy
    elif idx == 2:
        Uyy = 1 - (1 - mu) * (termo1_32 - 3 * y**2 * termo1_52) \
            - mu * (termo2_32 - 3 * y**2 * termo2_52)
        return Uyy

#Define a matriz jacobiana 4x4 utilizando as derivadas segundas do potencial
@njit
def Jac(x, y, mu):
    jacobian = np.zeros((4, 4))

    jacobian[0, 2] = 1
    jacobian[1, 3] = 1

    jacobian[2, 3] = 2
    jacobian[3, 2] = -2

    for i in range(2):
        for j in range(2):
            jacobian[2+i][j] = potential_der(x, y, mu, i+j)

    return jacobian

#Função f(x) = \pdv{U}{x}(x, 0), ou seja, o gradiente de U na direcao x com y = 0
@njit
def f(x, mu):
    r1_cubo = abs(x + mu)**3
    r2_cubo = abs(x - 1 + mu)**3
    
    termo1 = (1 - mu) * (x + mu) / r1_cubo
    termo2 = mu * (x - 1 + mu) / r2_cubo
    
    return x - termo1 - termo2

#Derivada f'(x) = \pdv[2]{U}{x}(x, 0), ou seja, a derivada segunda de U com relacao a x em y = 0
@njit()
def df(x, mu):
    r1_cubo = abs(x + mu)**3
    r2_cubo = abs(x - 1 + mu)**3
    
    # Note o sinal de MAIS! Quando y=0, Uxx sempre tem essa forma positiva.
    return 1 + 2 * (1 - mu) / r1_cubo + 2 * mu / r2_cubo


#Algoritmo de Newton-Raphson central
@njit()
def resolver(x0, mu, nome_ponto, tolerancia=1e-12, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x, mu)
        
        #Retorna se a precisao requerida ja foi atendida
        if abs(fx) < tolerancia:
            return x
        
        dfx = df(x, mu)
        #retorna erro se a derivada f'(x) zerou
        if dfx == 0:
            print(f"Erro: Derivada zerou no cálculo de {nome_ponto}.")
            return np.nan
            
        # Passo do Newton-Raphson: x_novo = x_antigo - f(x)/f'(x)
        x = x - fx / dfx
        
    print(f"Aviso: Limite de iterações atingido para {nome_ponto}.")
    return x

#Metodo de Newton-Raphson para encontrar os pontos fixos colineares
@njit()
def newton_raphson_lagrange(mu):
    #Chutes iniciais (Aproximações de Hill)
    alpha = (mu / 3)**(1/3) # este é um parametro relativamente pequeno
    
    #L1 fica entre as massas principais
    x0_L1 = 1 - mu - alpha
    
    #L2 fica a direita da massa menor
    x0_L2 = 1 - mu + alpha
    
    #L3 fica do lado oposto da massa maior
    x0_L3 = -(1 + (5/12) * mu)
    
    # Encontrando as raizes
    L1_x = resolver(x0_L1, mu, "L1")
    L2_x = resolver(x0_L2, mu, "L2")
    L3_x = resolver(x0_L3, mu, "L3")
    
    return np.array([L1_x, L2_x, L3_x])

#Derivada do potencial. aqui r e um array da forma [x, y]. dir e uma variavel que diz se o valor retornado e o gradiente na direcao x ou y
@njit()
def Inertial_pot_der(r, mu, dir):
    if dir == 0:
        return r[0] - (1-mu)*(mu + r[0])/(((mu+r[0])**2 + r[1]**2)**(3/2)) + mu*(1 - mu - r[0])/(((1-mu-r[0])**2 + r[1]**2)**(3/2))
    elif dir == 1:
        return r[1] - (1-mu)*r[1]/(((mu+r[0])**2 + r[1]**2)**(3/2)) - mu*r[1]/(((1-mu-r[0])**2 + r[1]**2)**(3/2))


#Funcao para gerar animacao das trajetorias orbitais
def Animate(N, h, mu, time_interval, simulation_data, filename = 'simulacao_output.mp4'):
    #num_frames = 150
    #num_pontos = N
    # Array no formato (150 frames, 50 pontos, 2 coordenadas [x, y])

    passo = int(time_interval/h) #Aqui eu defino um novo tamanho de passo para reduzir o custo da animacao
    dados_simulacao = simulation_data[::passo]

    #Configuração da Figura
    #Cores dos corpos. No artigo eu uso azul para denotar os corpos que iniciam proximos de L_4 e L_5 e vermelho para o resto
    cores = ['red'] * int(6*N/10) + ['blue'] * int(4*N/10)

    fig, ax = plt.subplots(figsize = (12, 8))

    #Passamos a lista de cores no parâmetro 'c'
    scat = ax.scatter(dados_simulacao[0, :, 0], dados_simulacao[0, :, 1], 
                    c=cores, s=30)
  
    ax.scatter(-mu, 0., s = 120, c = 'black')
    ax.scatter(1-mu, 0., s = 60, c = 'black')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # 3. Função de Atualização
    def update(frame):
        #'frame' agora atua como um índice de tempo (t) no array
        new_positions = dados_simulacao[frame]
        
        #set_offsets ja espera um array no formato (N, 2), 
        #que é exatamente o que temos se fatiarmos o array principal
        scat.set_offsets(new_positions)
        return (scat,)

    #Criar, Salvar e Exibir
    #O número de frames é o tamanho da primeira dimensão do seu array
    ani = FuncAnimation(fig, update, frames=len(dados_simulacao), interval=40, blit=True)
    ani.save(filename, writer='ffmpeg', fps=30, dpi=150)

    #Fechar o plot estático e exibir o player interativo no Jupyter
    plt.close()
    return HTML(ani.to_jshtml())

#Implementacao da rotina utilizando o metodo de Runge-Kutta de 4a ordem (RK4) para integrar as orbitas
@njit(parallel = True)
def RK4_Rotating(N, mu, r0, v0, T, h):
    nu = int((T+h)/h) #numero de passos no tempo

    #Arrays para armazenar a trajetória e a velocidade ao longo do tempo
    r = np.zeros((nu, N, 2))
    v = np.zeros((nu, N, 2))

    #Condições iniciais
    r[0] = r0
    v[0] = v0

    #Loop de integração temporal
    #Aqui o prange faz o papel da paralelizacao. Como as trajetorias dos N corpos sao independentes, podemos
    #calcularcada trajetoria separadamente de forma simultanea
    for j in prange(N):
        for i in range(nu-1):
            #Extraimos o estado atual da partícula 'i' no tempo 'j'
            r_atual = r[i, j, :]
            v_atual = v[i, j, :]

            #Passo 1 (k1): Derivadas no ponto inicial
            ax1 =  2 * v_atual[1] + Inertial_pot_der(r_atual, mu, 0)
            ay1 = -2 * v_atual[0] + Inertial_pot_der(r_atual, mu, 1)
            
            k1_r = v_atual
            k1_v = np.array([ax1, ay1])

            #Passo 2 (k2): Derivadas no ponto médio (usando k1)
            r2 = r_atual + 0.5 * h * k1_r
            v2 = v_atual + 0.5 * h * k1_v
            
            ax2 =  2 * v2[1] + Inertial_pot_der(r2, mu, 0)
            ay2 = -2 * v2[0] + Inertial_pot_der(r2, mu, 1)
            
            k2_r = v2
            k2_v = np.array([ax2, ay2])

            #Passo 3 (k3): Derivadas no ponto médio (usando k2)
            r3 = r_atual + 0.5 * h * k2_r
            v3 = v_atual + 0.5 * h * k2_v
            
            ax3 =  2 * v3[1] + Inertial_pot_der(r3, mu, 0)
            ay3 = -2 * v3[0] + Inertial_pot_der(r3, mu, 1)
            
            k3_r = v3
            k3_v = np.array([ax3, ay3])

            #Passo 4 (k4): Derivadas no final do intervalo (usando k3)
            r4 = r_atual + h * k3_r
            v4 = v_atual + h * k3_v
            
            ax4 =  2 * v4[1] + Inertial_pot_der(r4, mu, 0)
            ay4 = -2 * v4[0] + Inertial_pot_der(r4, mu, 1)
            
            k4_r = v4
            k4_v = np.array([ax4, ay4])

            #Atualizacao das posicoes e velocidades
            r[i+1, j, :] = r_atual + (h / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            v[i+1, j, :] = v_atual + (h / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r, v

#Funcao que implementa passos unicos para serem utilizados no metodo de Benettin
@njit()
def rk4_step_fast(r, v, mu, h):
    #Desempacota os arrays em variaveis escalares
    rx, ry = r[0], r[1]
    vx, vy = v[0], v[1]
    
    #Unico array alocado na memoria para servir de entrada na sua função do potencial
    r_temp = np.zeros(2)
    
    #Passo 1
    r_temp[0], r_temp[1] = rx, ry
    ax1 =  2 * vy + Inertial_pot_der(r_temp, mu, 0)
    ay1 = -2 * vx + Inertial_pot_der(r_temp, mu, 1)
    k1_rx, k1_ry = vx, vy
    k1_vx, k1_vy = ax1, ay1
    
    #Passo 2
    r_temp[0], r_temp[1] = rx + 0.5*h*k1_rx, ry + 0.5*h*k1_ry
    ax2 =  2 * (vy + 0.5*h*k1_vy) + Inertial_pot_der(r_temp, mu, 0)
    ay2 = -2 * (vx + 0.5*h*k1_vx) + Inertial_pot_der(r_temp, mu, 1)
    k2_rx, k2_ry = vx + 0.5*h*k1_vx, vy + 0.5*h*k1_vy
    k2_vx, k2_vy = ax2, ay2
    
    #Passo 3
    r_temp[0], r_temp[1] = rx + 0.5*h*k2_rx, ry + 0.5*h*k2_ry
    ax3 =  2 * (vy + 0.5*h*k2_vy) + Inertial_pot_der(r_temp, mu, 0)
    ay3 = -2 * (vx + 0.5*h*k2_vx) + Inertial_pot_der(r_temp, mu, 1)
    k3_rx, k3_ry = vx + 0.5*h*k2_vx, vy + 0.5*h*k2_vy
    k3_vx, k3_vy = ax3, ay3
    
    #Passo 4
    r_temp[0], r_temp[1] = rx + h*k3_rx, ry + h*k3_ry
    ax4 =  2 * (vy + h*k3_vy) + Inertial_pot_der(r_temp, mu, 0)
    ay4 = -2 * (vx + h*k3_vx) + Inertial_pot_der(r_temp, mu, 1)
    k4_rx, k4_ry = vx + h*k3_vx, vy + h*k3_vy
    k4_vx, k4_vy = ax4, ay4
    

    #So empacotamos de volta no final para devolver os novos r e v
    r_next = np.array([rx + (h/6.0)*(k1_rx + 2*k2_rx + 2*k3_rx + k4_rx),
                       ry + (h/6.0)*(k1_ry + 2*k2_ry + 2*k3_ry + k4_ry)])
    
    v_next = np.array([vx + (h/6.0)*(k1_vx + 2*k2_vx + 2*k3_vx + k4_vx),
                       vy + (h/6.0)*(k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)])
                       
    return r_next, v_next

#Implementacao do metodo de Benettin
@njit()
def benettin_mle(mu, r0, v0, T, h, d0=1e-8):
    steps = int(T / h)
    
    #Configurando a orbita Nominal
    r_ref = r0.copy()
    v_ref = v0.copy()
    
    #Configurando a orbita Perturbada
    #Criamos a perturbação no eixo x, o sistema cuidara do resto
    r_pert = r0.copy()
    r_pert[0] += d0
    v_pert = v0.copy()
    
    soma_log = 0.0
    
    for i in range(steps):
        #Avanca ambas as orbitas 1 passo no tempo
        r_ref_next, v_ref_next = rk4_step_fast(r_ref, v_ref, mu, h)
        r_pert_next, v_pert_next = rk4_step_fast(r_pert, v_pert, mu, h)
        
        #Calcula o vetor de separacao nas 4 coordenadas
        dr = r_pert_next - r_ref_next
        dv = v_pert_next - v_ref_next
        
        #Mede a distancia d1 no espaço de fase completo (de dimensao 4)
        d1 = np.sqrt(dr[0]**2 + dr[1]**2 + dv[0]**2 + dv[1]**2)
        
        #Prevencao contra colapso numerico (se as órbitas cairem exatamente no mesmo ponto)
        if d1 == 0:
            d1 = 1e-16
            
        #Acumula a taxa de expansao
        soma_log += np.log(d1 / d0)
        
        #Renormalizacao: Encolhe a distancia de volta para d0
        fator = d0 / d1
        r_pert = r_ref_next + dr * fator
        v_pert = v_ref_next + dv * fator
        
        #Atualiza a referencia para o proximo loop
        r_ref = r_ref_next
        v_ref = v_ref_next
        
    #O Expoente Maximo de Lyapunov é a média no tempo
    return soma_log / T

#Funcao que chama o algoritmo de Benettin para cada massa reduzida e cada ponto fixo
@njit(parallel = True)
def Lyapunov_Benettin_Scan(T, h, mu_arr):
    #Array para armazenar o MLE dos 5 pontos para cada mu
    Lyapunov_exp = np.zeros((len(mu_arr), 5))
    
    for k in prange(len(mu_arr)):
        mu = mu_arr[k]
        
        #Calcula os pontos colineares usando Newton_Raphson
        pontos_colineares = newton_raphson_lagrange(mu)

        #Nos pontos de Lagrange as partículas iniciam em repouso
        v0 = np.array([0.0, 0.0])

        L_points = np.array([
            [pontos_colineares[0], 0.0],
            [pontos_colineares[1], 0.0],
            [pontos_colineares[2], 0.0],
            [0.5 - mu,  np.sqrt(3.0)/2.0],
            [0.5 - mu, -np.sqrt(3.0)/2.0]
        ])
        
        for i in range(5):
            r0 = L_points[i]
            
            mle = benettin_mle(mu, r0, v0, T, h)
            
            Lyapunov_exp[k, i] = mle
            
    return Lyapunov_exp

#Funcao para calculo de regressao linear
#Isso teve de ser feito pois funcoes comuns (como numpy.polyfit) nao sao suportadas no numba
@njit()
def calc_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerador = np.sum((x - x_mean) * (y - y_mean))
    denominador = np.sum((x - x_mean)**2)
    return numerador / denominador

#Funcao que calcula o coeficiente de lyapunov medio para cada ponto fixo atraves de uma distriuicao de pontos iniciais
#Esta funcao tambem calcula os coeficientes teoricos atraves dos autovalores da matriz jacobiana
@njit(parallel = True)
def Lyapunov_cloud(N, T, h, mu_arr, prob_amp = 1e-5):
    # Matrizes de saída
    Lyapunov_exp = np.zeros((len(mu_arr), 5))
    Lyapunov_std = np.zeros((len(mu_arr), 5))
    Lyap_anal = np.zeros((len(mu_arr), 5))
    
    for k in range(len(mu_arr)):
        mu = mu_arr[k]
        pontos_colineares = newton_raphson_lagrange(mu) #Calculo das posicoes dos pontos colineares

        #Array com as posicoes de todos os pontos de lagrange
        L_points = np.array([
            [pontos_colineares[0], 0.0],
            [pontos_colineares[1], 0.0],
            [pontos_colineares[2], 0.0],
            [0.5 - mu,  np.sqrt(3.0)/2.0],
            [0.5 - mu, -np.sqrt(3.0)/2.0]
        ])
        
        for i in range(5):
            jac_mat = Jac(L_points[i, 0], L_points[i, 1], mu) #Matriz jacobiana
            jac_mat_complex = jac_mat.astype(np.complex128) #Passamos a matriz jacobiana para tipo complexo para o numba conseguir lidar

            vals = np.linalg.eigvals(jac_mat_complex) #Calculo dos autovalores da jacobiana

            Lyap_anal[k, i] = np.max(np.real(vals)) #Valor maximo da parte real dos autovalores
            #A semente aqui garante que as perturbacoes sejam iguais para todos os pontos
            np.random.seed(42) 
            
            r0 = np.zeros((N, 2))
            v0 = np.zeros((N, 2))
            
            #Gerando a distribuicao gaussiana de corpos próximas ao Ponto de Lagrange
            for pt in range(N):
                r0[pt] = L_points[i] + prob_amp * np.random.randn(2)

            #Integrando as trajetórias 
            r1, v1 = RK4_Rotating(N, mu, r0, v0, T, h)

            time_interval = 1.0
            passo = int(time_interval / h)

            #Pegando os dados de forma espacada
            dados_simulacao = r1[::passo]
            time_arr = np.linspace(0, T, len(dados_simulacao))

            deltas = np.zeros((len(dados_simulacao), N-1))

            #Calculando a distancia dos corpos (1 a N-1) em relacao a particula 0
            for pt in range(1, N):
                for step in range(len(dados_simulacao)):
                    dx = dados_simulacao[step, pt, 0] - dados_simulacao[step, 0, 0]
                    dy = dados_simulacao[step, pt, 1] - dados_simulacao[step, 0, 1]
                    deltas[step, pt-1] = np.sqrt(dx**2 + dy**2)

            ps = np.zeros(N-1)
            
            #Realizando o ajuste linear no log das distancias
            for j in range(N-1):
                ps[j] = calc_slope(time_arr, np.log(deltas[:, j]))

            Lyapunov_exp[k, i] = np.mean(ps)
            Lyapunov_std[k, i] = np.std(ps) 

    return Lyap_anal, Lyapunov_exp, Lyapunov_std