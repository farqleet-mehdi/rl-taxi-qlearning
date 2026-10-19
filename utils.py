import numpy as np
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym

ENV_ID = "Taxi-v4"   # the only place the environment name is defined


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


def train(env, agent, episodes=10000, seed=0):
    """Train the agent. Returns the total reward of every episode."""
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        total, done = 0, False
        while not done:
            a = agent.select_action(s)
            s2, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s2, terminated)
            s, total = s2, total + r
            done = terminated or truncated
        agent.decay_epsilon()
        rewards.append(total)
    return np.array(rewards)


def evaluate(env, agent, episodes=100):
    """Run the greedy policy (no exploration). Returns mean reward, std, mean steps."""
    rewards, steps = [], []
    for _ in range(episodes):
        s, _ = env.reset()
        total, n, done = 0, 0, False
        while not done:
            a = agent.select_action(s, greedy=True)
            s, r, term, trunc, _ = env.step(a)
            total += r
            n += 1
            done = term or trunc
        rewards.append(total)
        steps.append(n)
    return np.mean(rewards), np.std(rewards), np.mean(steps)


def random_baseline(env, episodes=100):
    """Reward of a policy that picks random actions. Used for comparison."""
    rewards = []
    for _ in range(episodes):
        env.reset()
        total, done = 0, False
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def moving_avg(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_curves(curves, title, path, w=100):
    """curves is a dict: {label: array_of_episode_rewards}."""
    plt.figure(figsize=(7, 4))
    for label, r in curves.items():
        plt.plot(moving_avg(r, w), label=label)
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (moving avg, window={w})")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_gif(agent, path="taxi_agent.gif", fps=4, seed=1):
    """Record one greedy episode and save it as a GIF."""
    env = make_env(render_mode="rgb_array")
    s, _ = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    while not done:
        s, r, term, trunc, _ = env.step(agent.select_action(s, greedy=True))
        frames.append(env.render())
        done = term or trunc
    env.close()
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return len(frames) - 1   # number of steps the agent took