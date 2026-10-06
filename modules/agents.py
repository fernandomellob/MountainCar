import numpy as np
from collections import defaultdict
import random

class BaseAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1,
                 epsilon_decay=1.0, epsilon_min=0.0):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Decaimento multiplicativo de epsilon ao fim de cada episódio.
        # Com epsilon_decay = 1.0 (padrão) o epsilon fica fixo, como na versão original.
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Tabela Q: mapeia estado discreto para um array de valores das ações
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))
        self.q_updates_count = 0 # Para métrica de custo computacional

    def act(self, state_discrete):
        # Política Epsilon-Greedy
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_discrete])

    def end_episode(self):
        """Chamado ao fim de cada episódio (decaimento de epsilon)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
    Agente com Propagação Acelerada usando Traços de Elegibilidade (traços acumulativos).

    Os traços ativos ficam em um dicionário esparso {(estado, ação): traço}. Traços abaixo de
    `trace_threshold` são descartados, então cada passo atualiza só os pares recentemente
    visitados, e não a Q-table inteira.
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1, lambd=0.9,
                 trace_threshold=1e-3, **base_kwargs):
        super().__init__(action_space, alpha, gamma, epsilon, **base_kwargs)
        self.lambd = lambd
        self.trace_threshold = trace_threshold
        self.e_traces = {}

    def reset_traces(self):
        self.e_traces.clear()

    def learn(self, state, action, reward, next_state, next_action, done):
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        td_error = td_target - self.q_table[state][action]

        key = (state, action)
        self.e_traces[key] = self.e_traces.get(key, 0.0) + 1.0

        # Atualiza todos os pares com traço ativo e decai os traços
        decay = self.gamma * self.lambd
        expired = []
        for (s, a), e in self.e_traces.items():
            self.q_table[s][a] += self.alpha * td_error * e
            self.q_updates_count += 1
            e *= decay
            if e < self.trace_threshold:
                expired.append((s, a))
            else:
                self.e_traces[(s, a)] = e
        for k in expired:
            del self.e_traces[k]

class DynaQAgent(BaseAgent):
    """
    Agente Baseado em Modelos: Realiza planejamento com modelo simulado
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=10,
                 **base_kwargs):
        super().__init__(action_space, alpha, gamma, epsilon, **base_kwargs)
        self.n_planning_steps = n_planning_steps
        self.model = {} # Mapeia (state, action) para (reward, next_state, done)
        self.model_keys = [] # Pares (state, action) já observados, para sorteio em O(1)

    def learn(self, state, action, reward, next_state, done):
        # 1. Atualização Q-Learning normal
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.q_updates_count += 1
        
        # 2. Atualização do Modelo (determinístico: guarda o último desfecho observado)
        key = (state, action)
        if key not in self.model:
            self.model_keys.append(key)
        self.model[key] = (reward, next_state, done)
            
        # 3. Planejamento (Planning)
        for _ in range(self.n_planning_steps):
            # Escolhe um par (estado, ação) já observado, uniformemente
            s, a = self.model_keys[random.randrange(len(self.model_keys))]
            r, next_s, d = self.model[(s, a)]
            
            best_a = np.argmax(self.q_table[next_s])
            p_td_target = r + self.gamma * self.q_table[next_s][best_a] * (not d)
            self.q_table[s][a] += self.alpha * (p_td_target - self.q_table[s][a])
            self.q_updates_count += 1
