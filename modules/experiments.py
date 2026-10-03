import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from tqdm import tqdm

def run_experiment(agent_type, discretizer, env_name="MountainCar-v0", episodes=500, seeds=[42], **agent_kwargs):
    """
    Roda um experimento para múltiplas seeds.
    Retorna um DataFrame estruturado para uso fácil no Seaborn.
    """
    results = []
    
    for seed in tqdm(seeds, desc=f"Treinando {agent_type.__name__}"):
        # Inicializa ambiente com seed fixa
        env = gym.make(env_name)
        np.random.seed(seed)
        env.action_space.seed(seed)
        
        agent = agent_type(env.action_space, **agent_kwargs)
        total_steps = 0
        
        for ep in range(episodes):
            state, _ = env.reset(seed=seed+ep) # Varia a seed inicial por episódio para exploração
            state_d = discretizer.discretize(state)
            
            done = False
            truncated = False
            steps = 0
            ep_reward = 0
            
            # Para SARSA lambda
            if hasattr(agent, 'reset_traces'):
                agent.reset_traces()
                
            action = agent.act(state_d)
            
            while not (done or truncated):
                next_state, reward, done, truncated, _ = env.step(action)
                next_state_d = discretizer.discretize(next_state)
                next_action = agent.act(next_state_d)
                
                if hasattr(agent, 'reset_traces'):
                    agent.learn(state_d, action, reward, next_state_d, next_action, done)
                else:
                    agent.learn(state_d, action, reward, next_state_d, done)
                    
                state_d = next_state_d
                action = next_action
                steps += 1
                total_steps += 1
                ep_reward += reward
                
            results.append({
                'Agent': agent_type.__name__,
                'Seed': seed,
                'Episode': ep,
                'Steps': steps,
                'Cumulative Steps': total_steps,
                'Q-Updates': agent.q_updates_count,
                'Reward': ep_reward
            })
            
        env.close()
        
    return pd.DataFrame(results)

def plot_learning_curves(df):
    """
    Gera os gráficos conforme exigido na Tarefa 4.1.
    """
    sns.set_theme(style="darkgrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Passos até o sucesso vs Episódios
    sns.lineplot(data=df, x='Episode', y='Steps', hue='Agent', ax=axes[0], errorbar=('ci', 95))
    axes[0].set_title("Desempenho (Steps por Episódio)")
    axes[0].set_ylabel("Passos no Ambiente")
    
    # Gráfico 2: Atualizações Q vs Episódios
    sns.lineplot(data=df, x='Episode', y='Q-Updates', hue='Agent', ax=axes[1], errorbar=('ci', 95))
    axes[1].set_title("Custo Computacional (Atualizações de Valor-Q)")
    axes[1].set_ylabel("Total de Updates Acumulados")
    
    plt.tight_layout()
    plt.show()