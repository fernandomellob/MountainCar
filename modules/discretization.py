import numpy as np

class StateDiscretizer:
    def __init__(self, env, bins_position, bins_velocity):
        """
        Inicializa o discretizador com base nos limites do ambiente e quantidade de caixas (bins).
        """
        self.env = env
        # Limites do ambiente MountainCar-v0
        self.pos_min, self.vel_min = env.observation_space.low
        self.pos_max, self.vel_max = env.observation_space.high
        
        # Criando os bins
        self.pos_bins = np.linspace(self.pos_min, self.pos_max, bins_position)
        self.vel_bins = np.linspace(self.vel_min, self.vel_max, bins_velocity)
        
    def discretize(self, state):
        """
        Converte um estado contínuo (posição, velocidade) em um estado discreto (tupla de índices).
        """
        # Trata caso o estado venha aninhado (comum em versões novas do Gymnasium)
        if isinstance(state, tuple):
            state = state[0]
            
        pos, vel = state
        pos_idx = np.digitize(pos, self.pos_bins) - 1
        vel_idx = np.digitize(vel, self.vel_bins) - 1
        
        return (pos_idx, vel_idx)

# Funções auxiliares para criar discretizações "grossas" e "finas"
def get_coarse_discretizer(env):
    # Exemplo: 10 bins para posição, 10 para velocidade
    return StateDiscretizer(env, bins_position=10, bins_velocity=10)

def get_fine_discretizer(env):
    # Exemplo: 40 bins para posição, 40 para velocidade
    return StateDiscretizer(env, bins_position=40, bins_velocity=40)