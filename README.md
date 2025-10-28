SimCQC: AI Simulation in Military Strategy (ICHack 2025 runner-up).

My Contributions (all commits made from team lead's computer):

- Implemented a statistics and analytics module for our CQC (close-quarters combat) simulation environment.
- Designed Python classes and methods to:
  - Track attacker/defender survival times and calculate “first blood” events.
  - Measure displacement, total distance travelled, and per-round movement of agents.
  - Compute spatial coverage and variance using agents’ field-of-view.
  - Analyse team coordination by calculating inter-attacker distances and spread variance.
- Exposed these results through a structured JSON API (statistics() method) for seamless integration with our reinforcement learning algorithm.
- Ensured the simulation could provide quantitative insights into combat strategies, enabling reinforcement learning agents to optimise decision-making.

This contribution helped the team demonstrate how AI-driven metrics can inform and improve tactical military strategies in simulated environments.
