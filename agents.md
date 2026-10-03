Estrutura do Projeto

Para manter o código organizado e facilitar a reutilização e testes, o projeto será dividido em módulos Python que serão importados no seu notebook principal.

Arquivos:

discretization.py: Conterá as funções/classes responsáveis por converter o espaço contínuo de observação do MountainCar-v0 em estados discretos (Tarefa A).

agents.py: Conterá as implementações dos agentes tabulares (Q-Learning/SARSA, SARSA(λ) e Dyna-Q) (Tarefa B).

experiments.py: Conterá as funções para orquestrar o treinamento com múltiplas sementes (seeds), coletar os dados e plotar os gráficos (Tarefas C e 4.1).

main_notebook.ipynb: Seu notebook base, onde você instanciará os objetos, rodará os experimentos e escreverá os relatórios.