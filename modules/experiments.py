import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from tqdm import tqdm

def run_experiment(agent_type, discretizer, env_name="MountainCar-v0", episodes=500, seeds=[42], representation_name="Grossa (10x10)", **agent_kwargs):
    """
    Roda um experimento para múltiplas seeds.
    Retorna um DataFrame estruturado para uso fácil no Seaborn.
    """
    results = []
    
    for seed in tqdm(seeds, desc=f"Treinando {agent_type.__name__} [{representation_name}]"):
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
                'Representation': representation_name,
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
    Gera os gráficos com Intervalo de Confiança de 95% (sombreamento) para os 3 algoritmos.
    Suporta visualização comparativa para múltiplas representações de estados (Grossa vs Fina).
    """
    sns.set_theme(style="darkgrid")
    
    if 'Representation' in df.columns and df['Representation'].nunique() > 1:
        representations = df['Representation'].unique()
        fig, axes = plt.subplots(2, len(representations), figsize=(8 * len(representations), 10))
        
        for i, rep in enumerate(representations):
            df_rep = df[df['Representation'] == rep]
            
            # Gráfico de Passos por Episódio (Desempenho / Eficiência Amostral)
            sns.lineplot(data=df_rep, x='Episode', y='Steps', hue='Agent', ax=axes[0, i], errorbar=('ci', 95))
            axes[0, i].set_title(f"Passos no Ambiente (Desempenho) - {rep}")
            axes[0, i].set_ylabel("Passos por Episódio")
            axes[0, i].set_xlabel("Episódio")
            
            # Gráfico de Q-Updates Acumulados (Custo / Eficiência Computacional)
            sns.lineplot(data=df_rep, x='Episode', y='Q-Updates', hue='Agent', ax=axes[1, i], errorbar=('ci', 95))
            axes[1, i].set_title(f"Custo Computacional (Q-Updates) - {rep}")
            axes[1, i].set_ylabel("Atualizações Q Acumuladas")
            axes[1, i].set_xlabel("Episódio")
            
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.lineplot(data=df, x='Episode', y='Steps', hue='Agent', ax=axes[0], errorbar=('ci', 95))
        axes[0].set_title("Desempenho (Steps por Episódio)")
        axes[0].set_ylabel("Passos no Ambiente")
        axes[0].set_xlabel("Episódio")
        
        sns.lineplot(data=df, x='Episode', y='Q-Updates', hue='Agent', ax=axes[1], errorbar=('ci', 95))
        axes[1].set_title("Custo Computacional (Atualizações de Valor-Q)")
        axes[1].set_ylabel("Total de Updates Acumulados")
        axes[1].set_xlabel("Episódio")
        
        plt.tight_layout()
        plt.show()

def summarize_experiments(df):
    """
    Retorna uma tabela resumo comparativa com estatísticas agregadas por Agente e Representação.
    """
    last_episodes = df[df['Episode'] >= df['Episode'].max() - 50]
    
    group_cols = ['Representation', 'Agent'] if 'Representation' in df.columns else ['Agent']
    
    summary = last_episodes.groupby(group_cols).agg(
        Mean_Final_Steps=('Steps', 'mean'),
        Std_Final_Steps=('Steps', 'std'),
        Mean_Total_Q_Updates=('Q-Updates', lambda x: df.loc[x.index].groupby('Seed')['Q-Updates'].max().mean()),
        Std_Total_Q_Updates=('Q-Updates', lambda x: df.loc[x.index].groupby('Seed')['Q-Updates'].max().std())
    ).reset_index()
    
    return summary