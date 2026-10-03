import numpy as np
from collections import defaultdict
import random

class BaseAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Tabela Q: mapeia estado discreto para um array de valores das ações
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))
        self.q_updates_count = 0 # Para métrica de custo computacional
        
    def act(self, state_discrete):
        # Política Epsilon-Greedy
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_discrete])

class QLearningAgent(BaseAgent):
    """
    Agente Baseline (TD 1-passo)
    """
    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.alpha * td_error
        self.q_updates_count += 1

class SarsaLambdaAgent(BaseAgent):
    """
    Agente com Propagação Acelerada usando Traços de Elegibilidade
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1, lambd=0.9):
        super().__init__(action_space, alpha, gamma, epsilon)
        self.lambd = lambd
        self.e_traces = defaultdict(lambda: np.zeros(self.action_space.n))

    def reset_traces(self):
        self.e_traces.clear()

    def learn(self, state, action, reward, next_state, next_action, done):
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        
        self.e_traces[state][action] += 1
        
        # Atualiza todos os estados baseados no traço de elegibilidade
        for s, q_values in self.q_table.items():
            for a in range(self.action_space.n):
                if self.e_traces[s][a] > 0:
                    self.q_table[s][a] += self.alpha * td_error * self.e_traces[s][a]
                    self.e_traces[s][a] *= self.gamma * self.lambd
                    self.q_updates_count += 1

class DynaQAgent(BaseAgent):
    """
    Agente Baseado em Modelos: Realiza planejamento com modelo simulado
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=10):
        super().__init__(action_space, alpha, gamma, epsilon)
        self.n_planning_steps = n_planning_steps
        self.model = {} # Mapeia (state, action) para (reward, next_state)
        self.visited_states = []

    def learn(self, state, action, reward, next_state, done):
        # 1. Atualização Q-Learning normal
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.q_updates_count += 1
        
        # 2. Atualização do Modelo
        self.model[(state, action)] = (reward, next_state)
        if state not in self.visited_states:
            self.visited_states.append(state)
            
        # 3. Planejamento (Planning)
        for _ in range(self.n_planning_steps):
            if not self.model:
                break
            # Escolhe estado e ação aleatórios já observados
            idx = random.randint(0, len(self.model) - 1)
            (s, a), (r, next_s) = list(self.model.items())[idx]
            
            best_a = np.argmax(self.q_table[next_s])
            p_td_target = r + self.gamma * self.q_table[next_s][best_a]
            self.q_table[s][a] += self.alpha * (p_td_target - self.q_table[s][a])
            self.q_updates_count += 1