from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation import Simulation, Map, Strategy, DEFAULT_ATTACKER_PARAMS, DEFAULT_DEFENDER_PARAMS

app = Flask(__name__)

# Enable CORS for all routes using flask_cors.
CORS(app)  # This will add the appropriate CORS headers automatically

@app.route("/simulate", methods=["POST", "OPTIONS"])
def simulate_endpoint():
    if request.method == "OPTIONS":
        # Preflight request: return an empty response with 200 OK.
        return jsonify({}), 200

    # For POST requests, force reading JSON.
    data = request.get_json(force=True)
    
    grid = data.get("grid", [[0 for _ in range(10)] for _ in range(10)])
    attacker_positions = data.get("attacker_positions", [(0, 0), (0, 9)])
    defender_positions = data.get("defender_positions", [(3, 7), (2, 7), (8, 6)])
    attacker_params = data.get("attacker_params", DEFAULT_ATTACKER_PARAMS)
    defender_params = data.get("defender_params", DEFAULT_DEFENDER_PARAMS)
    
    # Convert list positions to tuples.
    attacker_positions = [tuple(pos) for pos in attacker_positions]
    defender_positions = [tuple(pos) for pos in defender_positions]
    
    strategy = Strategy(attacker_positions)
    simulation = Simulation(Map(grid), strategy, defender_positions, attacker_params, defender_params)
    result = simulation.run()
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
