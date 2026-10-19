import numpy as np
from agent import QLearningAgent
from utils import (make_env, train, evaluate, random_baseline,
                   plot_curves, make_gif, ENV_ID)

# ---- Final hyperparameters (these go in the report table) ----
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999
SEED = 0
EVAL_EPISODES = 100


def main():
    env = make_env()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f"Environment: {ENV_ID} | states: {n_states} | actions: {n_actions}")

    # 1. Random baseline
    rb_mean, rb_std = random_baseline(env, EVAL_EPISODES)
    print(f"Random baseline: mean={rb_mean:.2f}, std={rb_std:.2f}")

    # 2. Train
    agent = QLearningAgent(n_states, n_actions, alpha=ALPHA, gamma=GAMMA,
                           eps_start=EPS_START, eps_end=EPS_END,
                           eps_decay=EPS_DECAY, seed=SEED)
    rewards = train(env, agent, EPISODES, seed=SEED)
    print(f"Training done. First 100 avg: {rewards[:100].mean():.2f} | "
          f"last 100 avg: {rewards[-100:].mean():.2f}")

    # 3. Evaluate the greedy policy
    mean, std, steps = evaluate(env, agent, EVAL_EPISODES)
    print(f"Greedy eval: mean reward={mean:.2f}, std={std:.2f}, mean steps={steps:.2f}")

    # 4. Save Q-table, raw rewards, and learning curve
    agent.save("models/q_table.npy")
    np.save("results/training_rewards.npy", rewards)
    plot_curves({"Q-learning": rewards}, f"{ENV_ID} Training Curve",
                "results/learning_curve.png")

    # 5. Save GIF of the trained agent
    n_steps = make_gif(agent, "taxi_agent.gif")
    print(f"GIF saved. Episode length: {n_steps} steps")

    # 6. Save a text summary (handy for the report)
    with open("results/summary.txt", "w") as f:
        f.write(f"Environment: {ENV_ID}\n")
        f.write(f"Episodes: {EPISODES}, alpha={ALPHA}, gamma={GAMMA}, "
                f"eps {EPS_START}->{EPS_END}, decay={EPS_DECAY}, seed={SEED}\n")
        f.write(f"Random baseline: mean={rb_mean:.2f}, std={rb_std:.2f}\n")
        f.write(f"Greedy eval ({EVAL_EPISODES} eps): mean={mean:.2f}, "
                f"std={std:.2f}, steps={steps:.2f}\n")
        f.write(f"GIF episode length: {n_steps} steps\n")


if __name__ == "__main__":
    main()