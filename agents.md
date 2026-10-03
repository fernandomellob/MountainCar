Estrutura do Projeto

Para manter o código organizado e facilitar a reutilização e testes, o projeto será dividido em módulos Python que serão importados no seu notebook principal.

Arquivos:

modules/discretization.py: Conterá as funções/classes responsáveis por converter o espaço contínuo de observação do MountainCar-v0 em estados discretos (Tarefa A).

modules/agents.py: Conterá as implementações dos agentes tabulares (Q-Learning/SARSA, SARSA(λ) e Dyna-Q) (Tarefa B).

modules/experiments.py: Conterá as funções para orquestrar o treinamento com múltiplas sementes (seeds), coletar os dados e plotar os gráficos (Tarefas C e 4.1).

RL_mountaiCart_project.ipynb: Seu notebook base, onde você instanciará os objetos, rodará os experimentos e escreverá os relatórios.