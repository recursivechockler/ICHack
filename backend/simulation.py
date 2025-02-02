import random
import math
import json
import statistics

# -------------------------
# Helper: Bresenham Line Algorithm
# -------------------------
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

# -------------------------
# Map / Building Class
# -------------------------
class Map:
    def __init__(self, grid):
        self.grid = grid  # grid: 0 = free; 1 = wall.
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def is_free(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == 0
        return False

    def get_random_free_cell(self):
        free_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.is_free(x, y)]
        if not free_cells:
            raise ValueError("No free cells available on the map.")
        return random.choice(free_cells)

# -------------------------
# Strategy Class for Attackers
# -------------------------
class Strategy:
    def __init__(self, attacker_positions):
        self.attacker_positions = attacker_positions

    def get_attacker_positions(self):
        return self.attacker_positions

# -------------------------
# Default Parameter Dictionaries
# -------------------------
DEFAULT_ATTACKER_PARAMS = {
    "vision_range": 5,
    "sound_radius": 4,
    "view_angle": math.pi / 4,
    "reaction": 1.0,
}
DEFAULT_DEFENDER_PARAMS = {
    "vision_range": 4,
    "sound_radius": 4,
    "view_angle": math.pi / 4,
    "reaction": 1.0,
}

# -------------------------
# Base Agent Class
# -------------------------
class Agent:
    def __init__(self, id, x, y, vision_range=3, orientation=0, view_angle=math.pi/4, sound_radius=3):
        self.id = id
        self.x = x
        self.y = y
        self.vision_range = vision_range
        self.orientation = orientation
        self.view_angle = view_angle
        self.sound_radius = sound_radius
        self.alive = True

    def position(self):
        return (self.x, self.y)

    def distance_to(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def line_of_sight_clear(self, other, game_map):
        line = bresenham_line(self.x, self.y, other.x, other.y)
        # Skip our own cell.
        for (cx, cy) in line[1:]:
            if not game_map.is_free(cx, cy):
                return False
        return True

    def can_see(self, other, game_map, epsilon=1e-6):
        if not other.alive:
            return False
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > self.vision_range:
            return False
        angle_to_other = math.atan2(dy, dx)
        angle_diff = abs((angle_to_other - self.orientation + math.pi) % (2 * math.pi) - math.pi)
        if angle_diff > self.view_angle + epsilon:
            return False
        if not self.line_of_sight_clear(other, game_map):
            return False
        return True

    def can_hear(self, other):
        if not other.alive:
            return False
        return self.distance_to(other) <= self.sound_radius

    def move_to(self, x, y):
        dx = x - self.x
        dy = y - self.y
        if dx or dy:
            self.orientation = math.atan2(dy, dx)
        self.x = x
        self.y = y

    def move_randomly(self, game_map):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_x = self.x + dx
            new_y = self.y + dy
            if game_map.is_free(new_x, new_y):
                self.move_to(new_x, new_y)
                break

    def move_towards(self, target_x, target_y, game_map):
        best_move = None
        best_dist = float("inf")
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_x = self.x + dx
            new_y = self.y + dy
            if game_map.is_free(new_x, new_y):
                dist = math.sqrt((target_x - new_x) ** 2 + (target_y - new_y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_move = (new_x, new_y)
        if best_move:
            self.move_to(*best_move)

# -------------------------
# RLAgent Base Class
# -------------------------
class RLAgent(Agent):
    ACTIONS = ["up", "down", "left", "right"]
    def __init__(self, id, x, y, vision_range, orientation, view_angle, sound_radius):
        super().__init__(id, x, y, vision_range, orientation, view_angle, sound_radius)
        self.starting_position = (x, y)
        self.q_values = {action: 0.0 for action in self.ACTIONS}
        self.last_action = None

    def choose_action(self, epsilon=0.1):
        if random.random() < epsilon:
            action = random.choice(self.ACTIONS)
        else:
            max_q = max(self.q_values.values())
            best_actions = [a for a, q in self.q_values.items() if q == max_q]
            action = random.choice(best_actions)
        self.last_action = action
        return action

    def update_q(self, reward, alpha=0.1, gamma=0.9):
        if self.last_action is not None:
            current_q = self.q_values[self.last_action]
            max_future_q = max(self.q_values.values())
            self.q_values[self.last_action] = current_q + alpha * (reward + gamma * max_future_q - current_q)

# -------------------------
# RLAttacker and RLDefender Classes
# -------------------------
class Attacker(RLAgent):
    def __init__(self, id, x, y, params={}):
        vision_range = params.get("vision_range", DEFAULT_ATTACKER_PARAMS["vision_range"])
        view_angle = params.get("view_angle", DEFAULT_ATTACKER_PARAMS["view_angle"])
        sound_radius = params.get("sound_radius", DEFAULT_ATTACKER_PARAMS["sound_radius"])
        orientation = params.get("orientation", random.uniform(0, 2 * math.pi))
        super().__init__(id, x, y, vision_range, orientation, view_angle, sound_radius)
        self.reaction = params.get("reaction", DEFAULT_ATTACKER_PARAMS["reaction"])
        self.visited = {(x, y)}
        self.score = 0

    def move_to(self, x, y):
        old_pos = self.position()
        super().move_to(x, y)
        # If new cell, add a positive reward; if not, impose a penalty.
        if (x, y) not in self.visited:
            self.visited.add((x, y))
            self.score += 1
        else:
            # This penalty is not used directly here; it is applied in perform_action.
            pass

    def perform_action(self, action, game_map, current_tick):
        """
        Perform an action and return a reward. This version heavily penalizes
        staying in previously visited cells. The penalty increases with time.
        """
        dx, dy = 0, 0
        if action == "up":
            dy = -1
        elif action == "down":
            dy = 1
        elif action == "left":
            dx = -1
        elif action == "right":
            dx = 1
        new_x = self.x + dx
        new_y = self.y + dy
        if game_map.is_free(new_x, new_y):
            # Compute Manhattan distance change from starting cell.
            old_dist = abs(self.x - self.starting_position[0]) + abs(self.y - self.starting_position[1])
            is_new = (new_x, new_y) not in self.visited
            self.move_to(new_x, new_y)
            new_dist = abs(self.x - self.starting_position[0]) + abs(self.y - self.starting_position[1])
            base_reward = new_dist - old_dist
            # Apply bonus if exploring new cell; otherwise heavy penalty.
            if is_new:
                reward = base_reward + 0.5
            else:
                # Penalty factor increases with current_tick; you can adjust the scaling factor (here 0.05).
                reward = base_reward - (5 + current_tick * 0.05)
            return reward
        else:
            # If attempting to move into a wall, return a fixed negative reward.
            return -10

class Defender(RLAgent):
    def __init__(self, id, x, y, params={}):
        vision_range = params.get("vision_range", DEFAULT_DEFENDER_PARAMS["vision_range"])
        view_angle = params.get("view_angle", DEFAULT_DEFENDER_PARAMS["view_angle"])
        sound_radius = params.get("sound_radius", DEFAULT_DEFENDER_PARAMS["sound_radius"])
        orientation = params.get("orientation", random.uniform(0, 2 * math.pi))
        super().__init__(id, x, y, vision_range, orientation, view_angle, sound_radius)
        self.reaction = params.get("reaction", DEFAULT_DEFENDER_PARAMS["reaction"])

    def perform_action(self, action, game_map):
        dx, dy = 0, 0
        if action == "up":
            dy = -1
        elif action == "down":
            dy = 1
        elif action == "left":
            dx = -1
        elif action == "right":
            dx = 1
        new_x = self.x + dx
        new_y = self.y + dy
        if game_map.is_free(new_x, new_y):
            old_dist = abs(self.x - self.starting_position[0]) + abs(self.y - self.starting_position[1])
            self.move_to(new_x, new_y)
            new_dist = abs(self.x - self.starting_position[0]) + abs(self.y - self.starting_position[1])
            reward = new_dist - old_dist
            return reward
        else:
            return -1

# -------------------------
# Simulation Class with Reinforcement Learning and Reset Capability
# -------------------------
class Simulation:
    def __init__(self, game_map, attacker_strategy, defender_positions,
                 attacker_params=None, defender_params=None, max_ticks=1000):
        self.map = game_map
        self.strategy = attacker_strategy
        self.attackers = []
        self.defenders = []
        self.tick = 0
        self.max_ticks = max_ticks

        if attacker_params is None:
            attacker_params = DEFAULT_ATTACKER_PARAMS.copy()
        if defender_params is None:
            defender_params = DEFAULT_DEFENDER_PARAMS.copy()

        attacker_positions = self.strategy.get_attacker_positions()
        for i, pos in enumerate(attacker_positions):
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
                orientation = pos[2] if len(pos) >= 3 else random.uniform(0, 2 * math.pi)
            else:
                raise ValueError("Invalid attacker position")
            params = attacker_params.copy()
            params["orientation"] = orientation
            self.attackers.append(Attacker(id=i, x=x, y=y, params=params))

        for i, pos in enumerate(defender_positions):
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
                orientation = pos[2] if len(pos) >= 3 else random.uniform(0, 2 * math.pi)
            else:
                raise ValueError("Invalid defender position")
            if any(a.x == x and a.y == y for a in self.attackers):
                raise ValueError(f"Defender starting cell ({x}, {y}) collides with an attacker.")
            params = defender_params.copy()
            params["orientation"] = orientation
            self.defenders.append(Defender(id=i, x=x, y=y, params=params))

        self.states = []

    def attackers_alive(self):
        return any(a.alive for a in self.attackers)

    def defenders_alive(self):
        return any(d.alive for d in self.defenders)

    def get_state(self):
        return {
            "tick": self.tick,
            "attackers": [
                {"id": a.id, "x": a.x, "y": a.y, "alive": a.alive, "score": a.score, "orientation": a.orientation}
                for a in self.attackers
            ],
            "defenders": [
                {"id": d.id, "x": d.x, "y": d.y, "alive": d.alive, "orientation": d.orientation}
                for d in self.defenders
            ]
        }

    def run(self):
        # Reset state for each run.
        self.states = []
        self.tick = 0
        self.states.append(self.get_state())
        while self.attackers_alive() and self.defenders_alive() and self.tick < self.max_ticks:
            self.tick += 1
            self.update()
            self.states.append(self.get_state())
        outcome = {"attackers_win": not self.defenders_alive()}
        stats = Statistics(self.states, self.map)
        return {"map": self.map.grid, "states": self.states, "outcome": outcome, "stats": stats.stats()}

    def update(self):
        engaged_attackers = set()
        engaged_defenders = set()

        # --- Engagement Phase ---
        for attacker in self.attackers:
            if not attacker.alive or attacker.id in engaged_attackers:
                continue
            visible_defenders = [d for d in self.defenders if d.alive and d.id not in engaged_defenders and attacker.can_see(d, self.map)]
            if visible_defenders:
                defender = min(visible_defenders, key=lambda d: attacker.distance_to(d))
                if defender.can_see(attacker, self.map):
                    a_reaction = random.uniform(0, attacker.reaction)
                    d_reaction = random.uniform(0, defender.reaction)
                    if a_reaction < d_reaction:
                        defender.alive = False
                        attacker.score += 5
                    else:
                        attacker.alive = False
                        attacker.score -= 10
                    engaged_attackers.add(attacker.id)
                    engaged_defenders.add(defender.id)
                else:
                    defender.alive = False
                    attacker.score += 5
                    engaged_attackers.add(attacker.id)
                    engaged_defenders.add(defender.id)
        for defender in self.defenders:
            if not defender.alive or defender.id in engaged_defenders:
                continue
            visible_attackers = [a for a in self.attackers if a.alive and a.id not in engaged_attackers and defender.can_see(a, self.map)]
            if visible_attackers:
                attacker = min(visible_attackers, key=lambda a: defender.distance_to(a))
                if not attacker.can_see(defender, self.map):
                    attacker.alive = False
                    attacker.score -= 10
                    engaged_defenders.add(defender.id)
                    engaged_attackers.add(attacker.id)
        # --- Movement Phase using RL ---
        for attacker in self.attackers:
            if not attacker.alive:
                continue
            # If any defender is in view, skip movement (engagement takes priority).
            if any(attacker.can_see(d, self.map) for d in self.defenders if d.alive):
                continue
            action = attacker.choose_action(epsilon=0.1)
            # Pass current tick so that the penalty for not exploring can increase over time.
            reward = attacker.perform_action(action, self.map, self.tick)
            attacker.update_q(reward, alpha=0.1, gamma=0.9)
        for defender in self.defenders:
            if not defender.alive:
                continue
            action = defender.choose_action(epsilon=0.1)
            reward = defender.perform_action(action, self.map)
            defender.update_q(reward, alpha=0.1, gamma=0.9)

# -------------------------
# Statistics Class
# -------------------------
class Statistics:
    def __init__(self, states, map_obj):
        self.states = states
        self.attackers = len(self.states[0]["attackers"])
        self.map = map_obj

    def unique_cells_visited(self, id):
        coords = []
        for state in self.states:
            x_cur = state["attackers"][id]["x"]
            y_cur = state["attackers"][id]["y"]
            if [x_cur, y_cur] not in coords:
                coords.append([x_cur, y_cur])
        return len(coords)

    def rounds_survived(self, index):
        rounds = 0
        for state in self.states:
            if state["attackers"][index]["alive"]:
                rounds += 1
            else:
                break
        return rounds

    def displacement(self, id):
        x_old = self.states[0]["attackers"][id]["x"]
        x_new = self.states[-1]["attackers"][id]["x"]
        y_old = self.states[0]["attackers"][id]["y"]
        y_new = self.states[-1]["attackers"][id]["y"]
        return abs(x_new - x_old) + abs(y_new - y_old)

    def first_blood_defender(self):
        tick = 1
        while tick <= len(self.states):
            for d in self.states[tick - 1]["defenders"]:
                if not d["alive"]:
                    return tick
            tick += 1
        return -1

    def first_blood_attacker(self):
        tick = 1
        while tick <= len(self.states):
            for a in self.states[tick - 1]["attackers"]:
                if not a["alive"]:
                    return tick
            tick += 1
        return -1

    def coverage_statistics(self):
        positions = set()
        def viewable_squares(x, y, angle, alive):
            if not alive:
                return
            rows = self.map.height
            cols = self.map.width
            if not (0 <= x < cols and 0 <= y < rows):
                return
            normalized_angle = angle % (2 * math.pi)
            if math.isclose(normalized_angle, 0) or math.isclose(normalized_angle, 2 * math.pi):
                for i in range(x + 1, cols):
                    if self.map.grid[y][i] == 1:
                        break
                    positions.add((y, i))
            elif math.isclose(normalized_angle, math.pi / 2):
                for j in range(y + 1, rows):
                    if self.map.grid[j][x] == 1:
                        break
                    positions.add((j, x))
            elif math.isclose(normalized_angle, math.pi):
                for i in range(x - 1, -1, -1):
                    if self.map.grid[y][i] == 1:
                        break
                    positions.add((y, i))
            elif math.isclose(normalized_angle, 3 * math.pi / 2):
                for j in range(y - 1, -1, -1):
                    if self.map.grid[j][x] == 1:
                        break
                    positions.add((j, x))
        coverages = []
        for state in self.states:
            for a in state["attackers"]:
                viewable_squares(a["x"], a["y"], a["orientation"], a["alive"])
            coverages.append(len(positions))
            positions = set()
        return (statistics.mean(coverages[:-1]), statistics.variance(coverages[:-1]))

    def total_distance_travelled(self, id):
        x0 = self.states[0]["attackers"][id]["x"]
        y0 = self.states[0]["attackers"][id]["y"]
        x1, y1 = x0, y0
        distance = 0
        time = 1
        while time < len(self.states) and self.states[time]["attackers"][id]["alive"]:
            x0, y0 = x1, y1
            x1 = self.states[time]["attackers"][id]["x"]
            y1 = self.states[time]["attackers"][id]["y"]
            time += 1
            distance += abs(x1 - x0) + abs(y1 - y0)
        return distance

    def distance_between_attacker(self):
        from math import sqrt
        total_dist = 0
        total_dist_squared = 0
        num_dists = 0
        for state in self.states:
            attackers = state["attackers"]
            for i in range(len(attackers) - 1):
                if attackers[i]["alive"]:
                    for j in range(i + 1, len(attackers)):
                        if attackers[j]["alive"]:
                            dx = attackers[i]["x"] - attackers[j]["x"]
                            dy = attackers[i]["y"] - attackers[j]["y"]
                            d = sqrt(dx * dx + dy * dy)
                            total_dist += d
                            total_dist_squared += d * d
                            num_dists += 1
        if num_dists == 0:
            return (-1, -1)
        avg = total_dist / num_dists
        avg2 = total_dist_squared / num_dists
        return (avg, avg2 - (avg ** 2))

    def stats(self):
        a = {
            "attacker_first_blood": self.first_blood_attacker(),
            "defender_first_blood": self.first_blood_defender(),
            "coverage_avg": self.coverage_statistics()[0],
            "coverage_var": self.coverage_statistics()[1],
        }
        spread_avg, spread_var = self.distance_between_attacker()
        a["spread_avg"] = spread_avg
        a["spread_var"] = spread_var
        distance_travelled = []
        displacement = []
        rounds_survived = []
        for i in range(self.attackers):
            distance_travelled.append(self.total_distance_travelled(i))
            displacement.append(self.displacement(i))
            rounds_survived.append(self.rounds_survived(i))
        a["distance_travelled_avg"] = statistics.mean(distance_travelled)
        a["distance_travelled_var"] = statistics.variance(distance_travelled)
        a["displacement_avg"] = statistics.mean(displacement)
        a["displacement_var"] = statistics.variance(displacement)
        a["rounds_survived_avg"] = statistics.mean(rounds_survived)
        a["rounds_survived_var"] = statistics.variance(rounds_survived)
        a["mean_unique_cells_visited"] = statistics.mean(
            [self.unique_cells_visited(i) for i in range(self.attackers)]
        )
        return a

# -------------------------
# Main Execution Example (for testing)
# -------------------------
if __name__ == "__main__":
    grid = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,1,1,0,0],
        [0,0,0,1,0,0,1,0,0,0],
        [0,1,1,1,0,0,1,0,0,0],
        [0,0,0,1,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    game_map = Map(grid)
    attacker_strategy = Strategy(attacker_positions=[(2,0), (7,0), (0,2), (9,5), (2,9), (7,9)])
    defender_positions = [(3,7), (2,7), (8,6)]
    simulation = Simulation(game_map, attacker_strategy, defender_positions,
                            attacker_params=DEFAULT_ATTACKER_PARAMS,
                            defender_params=DEFAULT_DEFENDER_PARAMS,
                            max_ticks=1000)
    result = simulation.run()
    print(json.dumps(result, indent=2))
