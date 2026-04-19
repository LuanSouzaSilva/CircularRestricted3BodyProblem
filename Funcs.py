#Import das bibliotecas que serão usadas. Como no último EP, a biblioteca NUMBA será utilizada para 
#transformar a função principal (que usa o algoritmo de Verlet) em linguagem compilada, acelerando sua execução
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#Aqui, por simplicidade, tanto matemática como física, já que ficará mais fácil interpretar os resultados,
#usei unidades tais que a massa solar, a unidade astronomica e o ano terrestre valem 1. Desta forma, 
#a constante gravitacional G vale G = 4pi^2, e a massa terrestre vale aproximadamente 3*10^-6 Msol
G = 4*(np.pi**2)
M = 1

#Algoritmo de Verlet para o cálculo da evolução temporal
@njit()
def Verlet(N, m_arr, r0, v0, T, h):
    nu = int((T+h)/h)

    r = np.zeros((nu, N, 3))
    v = np.zeros((nu, N, 3))

    r[0] = r0
    v[0] = v0

    #Como o método de Verlet é um algoritmo de passo duplo, usei o método de Euler-Cromer para calcular o r_1 a partir do r_0
    for j in range(N):
        a = np.zeros(3)
        for k in range(N):
            if j!=k:
                a += -G*m_arr[k]*(r[0][j] - r[0][k])/(np.linalg.norm(r[0][j] - r[0][k])**3)
            a += -G*M*(r[0][j])/(np.linalg.norm(r[0][j])**3)
        #Este if garante que o "Sol" permanecerá na origem
        if j != 0:
            v[1][j] = v[0][j] + a*h
            r[1][j] = r[0][j] + v[1][j]*h

    #Aqui é feito o algoritmo de verlet normalmente
    for i in range(1, nu-1):
        for j in range(N):
            a = np.zeros(3)
            for k in range(N):
                if j!=k:
                    a += -G*m_arr[k]*(r[i][j] - r[i][k])/(np.linalg.norm(r[i][j] - r[i][k])**3)
            if j != 0:
                r[i+1][j] = 2*r[i][j] - r[i-1][j] + a*(h**2)

    v1 = np.zeros((nu-1, N, 3))

    #Cálculo aproximado da velocidade como uma diferença finita da posição
    for i in range(1, nu):
        for j in range(N):
            v1[i-1][j] = (r[i][j] - r[i-1][j])/h

    return r, v1



#Função auxiliar que calcula a energia dos planetas 1 e 2 não perturbados,
#ou seja, sem considerar a interação entre os planetas.
#Esta função será usada para decidir se houve escape ou não, já que sabemos que
#a órbita é fechada se a energia E<0, caso contrário esta segue uma órbita parabólica ou hiperbólica.
#Claro que esta é uma forma aproximada de resolver o problema, mas foi a forma que eu pensei, e aparentemente funciona muito bem
def Energia(m1, m2, r1, r2, v1, v2):
    T1 = m1*(np.linalg.norm(v1, axis = 1)**2)/2
    V1 = -G*M*m1/np.linalg.norm(r1, axis = 1)

    T2 = m2*(np.linalg.norm(v2, axis = 1)**2)/2
    V2 = -G*M*m2/np.linalg.norm(r2, axis = 1)

    #Aqui ignoramos o último elemento do potencial pois, devido ao modo como a velocidade foi calculada, o numero de elementos do array é
    #um a menos que o array das posições
    E1 = T1 + V1[-len(V1)+1:]
    E2 = T2 + V2[-len(V2)+1:]

    return E1, E2

    

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

# Função f(x) = Ux(x, 0)
@njit
def f(x, mu):
    r1_cubo = abs(x + mu)**3
    r2_cubo = abs(x - 1 + mu)**3
    
    termo1 = (1 - mu) * (x + mu) / r1_cubo
    termo2 = mu * (x - 1 + mu) / r2_cubo
    
    return x - termo1 - termo2

# Derivada f'(x) = Uxx(x, 0)
@njit()
def df(x, mu):
    r1_cubo = abs(x + mu)**3
    r2_cubo = abs(x - 1 + mu)**3
    
    # Note o sinal de MAIS! Quando y=0, Uxx sempre tem essa forma positiva.
    return 1 + 2 * (1 - mu) / r1_cubo + 2 * mu / r2_cubo


# Algoritmo de Newton-Raphson central
@njit()
def resolver(x0, mu, nome_ponto, tolerancia=1e-12, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x, mu)
        
        # Se já chegamos perto o suficiente de zero, encerramos
        if abs(fx) < tolerancia:
            return x
        
        dfx = df(x, mu)
        if dfx == 0:
            print(f"Erro: Derivada zerou no cálculo de {nome_ponto}.")
            return np.nan
            
        # Passo do Newton-Raphson: x_novo = x_antigo - f(x)/f'(x)
        x = x - fx / dfx
        
    print(f"Aviso: Limite de iterações atingido para {nome_ponto}.")
    return x

@njit()
def newton_raphson_lagrange(mu):
    # --- CHUTES INICIAIS (Aproximações de Hill) ---
    # Estes são palpites clássicos da mecânica celeste que garantem a convergência
    alpha = (mu / 3)**(1/3)
    
    # L1 fica entre a massa primária e secundária (um pouco à esquerda da massa menor)
    x0_L1 = 1 - mu - alpha
    
    # L2 fica à direita da massa menor
    x0_L2 = 1 - mu + alpha
    
    # L3 fica do lado oposto da massa maior
    x0_L3 = -(1 + (5/12) * mu)
    
    # Encontrando as raízes
    L1_x = resolver(x0_L1, mu, "L1")
    L2_x = resolver(x0_L2, mu, "L2")
    L3_x = resolver(x0_L3, mu, "L3")
    
    return np.array([L1_x, L2_x, L3_x])

@njit()
def Inertial_pot_der(r, mu, dir):
    if dir == 0:
        return r[0] - (1-mu)*(mu + r[0])/(((mu+r[0])**2 + r[1]**2)**(3/2)) + mu*(1 - mu - r[0])/(((1-mu-r[0])**2 + r[1]**2)**(3/2))
    elif dir == 1:
        return r[1] - (1-mu)*r[1]/(((mu+r[0])**2 + r[1]**2)**(3/2)) - mu*r[1]/(((1-mu-r[0])**2 + r[1]**2)**(3/2))


@njit(parallel = True)
def Verlet_Inertial(N, mu, r0, v0, T, h):
    nu = int((T+h)/h)

    r = np.zeros((nu, N, 2))
    v = np.zeros((nu, N, 2))

    r[0] = r0
    v[0] = v0

    # Euler-Cromer para o primeiro passo
    a = np.zeros((2))
    for i in range(N):
        # CORREÇÃO CORIOLIS: a_x usa v_y (índice 1), a_y usa -v_x (índice 0)
        a[0] =  2 * v[0, i, 1] + Inertial_pot_der(r[0, i, :], mu, 0)
        a[1] = -2 * v[0, i, 0] + Inertial_pot_der(r[0, i, :], mu, 1)

        v[1, i, :] = v[0, i, :] + a*h
        r[1, i, :] = r[0, i, :] + v[1, i, :]*h

    # Algoritmo de Verlet
    for i in prange(N):
        for j in range(1, nu-1):
            a = np.zeros((2))

            # CORREÇÃO CORIOLIS: a_x usa v_y (índice 1), a_y usa -v_x (índice 0)
            a[0] =  2 * v[j, i, 1] + Inertial_pot_der(r[j, i, :], mu, 0)
            a[1] = -2 * v[j, i, 0] + Inertial_pot_der(r[j, i, :], mu, 1)

            r[j+1, i, :] = 2*r[j, i, :] - r[j-1, i, :] + a*(h**2)
            
            # Cálculo atrasado de velocidade (aviso: introduz erro numérico com o tempo)
            v[j+1, i] = (r[j+1, i, :] - r[j, i, :])/h

    return r, v


def Animate(N, h, mu, time_interval, simulation_data, filename = 'simulacao_output.mp4'):
    # ---------------------------------------------------------
    # 1. Simulação dos seus dados reais
    # ---------------------------------------------------------
    # Estou gerando um array falso aqui apenas para o código rodar.
    # Substitua 'dados_simulacao' pelo seu array NumPy real.
    #num_frames = 150
    #num_pontos = N
    # Array no formato (150 frames, 50 pontos, 2 coordenadas [x, y])

    passo = int(time_interval/h)

    dados_simulacao = simulation_data[::passo]


    # ---------------------------------------------------------
    # 2. Configuração da Figura
    # ---------------------------------------------------------

    # Pegamos os dados do instante t=0 para desenhar o primeiro frame
    posicoes_iniciais = dados_simulacao[0] # Formato: (50, 2)
    x_inicial = posicoes_iniciais[:, 0]
    y_inicial = posicoes_iniciais[:, 1]


    cores = ['red'] * int(6*N/10) + ['blue'] * int(4*N/10)

    fig, ax = plt.subplots(figsize = (12, 8))

    # Passamos a lista de cores no parâmetro 'c'
    scat = ax.scatter(dados_simulacao[0, :, 0], dados_simulacao[0, :, 1], 
                    c=cores, s=30)
  
    ax.scatter(-mu, 0., s = 120, c = 'black')
    ax.scatter(1-mu, 0., s = 60, c = 'black')
    #ax.gca().set_aspect('equal')
    #scat = ax.scatter(x_inicial, y_inicial, c='purple', alpha=0.7)

    # É muito importante fixar os limites dos eixos com base em TODOS os dados.
    # Isso impede que o gráfico fique mudando de escala durante a animação.
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)


    # ---------------------------------------------------------
    # 3. Função de Atualização
    # ---------------------------------------------------------
    def update(frame):
        # 'frame' agora atua como um índice de tempo (t) no seu array
        new_positions = dados_simulacao[frame]
        
        # set_offsets já espera um array no formato (N, 2), 
        # que é exatamente o que temos se fatiarmos o array principal
        scat.set_offsets(new_positions)
        return (scat,)


    # ---------------------------------------------------------
    # 4. Criar, Salvar e Exibir
    # ---------------------------------------------------------
    # O número de frames é o tamanho da primeira dimensão do seu array
    ani = FuncAnimation(fig, update, frames=len(dados_simulacao), interval=40, blit=True)
    ani.save(filename, writer='ffmpeg', fps=30, dpi=150)

    # Para salvar o arquivo físico, basta descomentar a linha abaixo:
    # ani.save('simulacao_dados_reais.mp4', writer='ffmpeg', fps=30)

    # Fechar o plot estático e exibir o player interativo no Jupyter
    plt.close()
    return HTML(ani.to_jshtml())



@njit(parallel = True)
def RK4_Rotating(N, mu, r0, v0, T, h):
    nu = int((T+h)/h)

    # Matrizes para armazenar a trajetória e a velocidade ao longo do tempo
    r = np.zeros((nu, N, 2))
    v = np.zeros((nu, N, 2))

    # Condições iniciais
    r[0] = r0
    v[0] = v0

    # Loop de integração temporal
    for j in prange(N):
        for i in range(nu-1):
            # Extraímos o estado atual da partícula 'i' no tempo 'j'
            r_atual = r[i, j, :]
            v_atual = v[i, j, :]

            # --- PASSO 1 (k1): Derivadas no ponto inicial ---
            ax1 =  2 * v_atual[1] + Inertial_pot_der(r_atual, mu, 0)
            ay1 = -2 * v_atual[0] + Inertial_pot_der(r_atual, mu, 1)
            
            k1_r = v_atual
            k1_v = np.array([ax1, ay1])

            # --- PASSO 2 (k2): Derivadas no ponto médio (usando k1) ---
            r2 = r_atual + 0.5 * h * k1_r
            v2 = v_atual + 0.5 * h * k1_v
            
            ax2 =  2 * v2[1] + Inertial_pot_der(r2, mu, 0)
            ay2 = -2 * v2[0] + Inertial_pot_der(r2, mu, 1)
            
            k2_r = v2
            k2_v = np.array([ax2, ay2])

            # --- PASSO 3 (k3): Derivadas no ponto médio (usando k2) ---
            r3 = r_atual + 0.5 * h * k2_r
            v3 = v_atual + 0.5 * h * k2_v
            
            ax3 =  2 * v3[1] + Inertial_pot_der(r3, mu, 0)
            ay3 = -2 * v3[0] + Inertial_pot_der(r3, mu, 1)
            
            k3_r = v3
            k3_v = np.array([ax3, ay3])

            # --- PASSO 4 (k4): Derivadas no final do intervalo (usando k3) ---
            r4 = r_atual + h * k3_r
            v4 = v_atual + h * k3_v
            
            ax4 =  2 * v4[1] + Inertial_pot_der(r4, mu, 0)
            ay4 = -2 * v4[0] + Inertial_pot_der(r4, mu, 1)
            
            k4_r = v4
            k4_v = np.array([ax4, ay4])

            # --- ATUALIZAÇÃO FINAL: Média ponderada dos 4 passos ---
            r[i+1, j, :] = r_atual + (h / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            v[i+1, j, :] = v_atual + (h / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return r, v