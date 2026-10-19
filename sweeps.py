import csv
import numpy as np
from agent import QLearningAgent
from utils import make_env, train, evaluate, plot_curves

BASE = dict(alpha=0.1, gamma=0.99, eps_decay=0.999)   # same as main.py
SEEDS = (0, 1, 2)
EPISODES = 5000
EVAL_EPISODES = 100

GRID = {
    "alpha": [0.01, 0.1, 0.5, 0.9],
    "gamma": [0.5, 0.9, 0.99],
    "eps_decay": [0.99, 0.999, 0.9999],
}


def run_setting(param, value):
    """Train with one parameter changed, over several seeds. Returns averaged results."""
    curves, eval_rewards, eval_steps = [], [], []
    for sd in SEEDS:
        env = make_env()
        kwargs = dict(BASE)
        kwargs[param] = value
        agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                               seed=sd, **kwargs)
        rewards = train(env, agent, EPISODES, seed=sd)
        mean, _, steps = evaluate(env, agent, EVAL_EPISODES)
        curves.append(rewards)
        eval_rewards.append(mean)
        eval_steps.append(steps)
    mean_curve = np.mean(curves, axis=0)
    return mean_curve, np.mean(eval_rewards), np.mean(eval_steps)


def main():
    rows = []
    for param, values in GRID.items():
        curves = {}
        for v in values:
            print(f"Running {param}={v} ...")
            curve, ev_reward, ev_steps = run_setting(param, v)
            curves[f"{param}={v}"] = curve
            rows.append([param, v, round(float(curve[-100:].mean()), 2),
                         round(float(ev_reward), 2), round(float(ev_steps), 2)])
        plot_curves(curves, f"Sensitivity to {param}", f"results/sensitivity_{param}.png")

    with open("results/sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "train_last100_avg",
                    "greedy_eval_reward", "greedy_eval_steps"])
        w.writerows(rows)

    print("\nparameter | value | train_last100 | greedy_reward | greedy_steps")
    for r in rows:
        print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()