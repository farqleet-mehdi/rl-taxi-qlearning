# Q-Learning Agent for Taxi-v3 (Gymnasium)

Custom reinforcement learning agent implemented from scratch using Python and
Farama Gymnasium. The agent learns to solve the Taxi-v3 environment using
tabular Q-learning with epsilon-greedy exploration.

**Course:** Topics in AI, COMSATS University, Attock Campus
**Assignment:** No. 1 (CLO-1)

## Project Structure

- `agent.py`: Q-learning agent (Q-table, epsilon-greedy policy, update rule)
- `utils.py`: training loop, evaluation, plotting, GIF rendering
- `main.py`: runs training, saves the Q-table, plots, and GIF
- `notebook.ipynb`: executed notebook with training progress and visualizations
- `models/`: saved Q-table (`q_table.npy`)
- `results/`: learning curve and hyperparameter sensitivity plots
- `requirements.txt`: dependencies

## Setup

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

    python main.py

## Results

| Policy | Mean reward | Std | Mean steps |
|---|---|---|---|
| Random baseline | -780.16 | 85.62 | n/a (hits 200-step limit) |
| Trained Q-learning (greedy, 100 eps) | 7.53 | 2.43 | 13.47 |

- Training: average reward went from -756.53 (first 100 episodes) to 4.58 (last 100) over 10,000 episodes.
- Sensitivity sweep results are in `results/sweep_results.csv` and the `results/sensitivity_*.png` plots.
- Demo of the trained agent: `taxi_agent.gif`.