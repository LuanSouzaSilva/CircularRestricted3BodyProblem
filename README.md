# Problema Circular de Três Corpos Restrito

Neste repositório estão os códigos usados para o trabalho da disciplina de pós-graduação Sistemas Caóticos Dinâmicos (PGF5433-1/1) no Instituto de Física da Universidade de São Paulo (IFUSP).

Os códigos e documentos estão organizados da seguinte forma:
1. Em $\verb|Funcs.py|$ estão todas as funções principais usadas para cálculos numéricos (Runge-Kutta, matriz jacobiana, derivadas do potencial efetivo, etc).

2. Em $\verb|Fixed_point_analysis.ipynb|$ estão os plots do espectro de autovalores da jacobiana e a análise de estabilidade dos pontos fixos

3. Em $\verb|Lyapunov.ipynb|$ estão os cálculos dos expoentes de Lyapunov do problema

4. Na pasta figuras estão as figuras salvas

5. Na pasta Animacao estão as animacoes das trajetorias

Os códigos usados neste trabalho não foram otimizados à exaustão, apesar de ter sido feito um bom esforço para tal, com várias partes tendo sido transformadas em código compilado (just-in-time), sendo que para isso foi preciso criar certas funções que já existem no numpy do zero, como a regressão linear para o ajuste no cálculo do expoente de Lyapunov médio. Além disso vários trechos de código foram paralelizados.

Por fim, a descrição do problema estará salva neste repositório (em português). Qualquer dúvida e/ou sugestão, pode me contatar via email por luansouzasilva@usp.br :D
