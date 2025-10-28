SimCQC: AI Simulation in Military Strategy (ICHack 2025 runner-up).

My Contributions (all commits made from team lead's computer):

- Architected and implemented the core reinforcement learning (Q-learning) engine for the simulation's autonomous agents.
- Designed and coded the RLAgent base class and the specific Attacker and Defender agent classes.
- Implemented the complete Q-learning algorithm from scratch, including:
    - Q-tables to store and update state-action values for each agent.
    - An epsilon-greedy policy (choose_action) to balance exploration and exploitation.
    - The core update_q function to allow agents to learn from rewards.
- Engineered a custom reward function (perform_action) for the Attacker agents that incentivised exploration by applying a time-increasing penalty for revisiting cells.
- Helped in developing the main Simulation class and training loop (run_training, run_episode) to manage agent training over multiple episodes and handle the simulation state.

This contribution helped the team demonstrate how AI-driven metrics can inform and improve tactical military strategies in simulated environments.
