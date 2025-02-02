from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation import Simulation, Map, Strategy, DEFAULT_ATTACKER_PARAMS, DEFAULT_DEFENDER_PARAMS
from anthropic import Anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

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

@app.route("/ask-claude", methods=["GET"])
def ask_claude():
    prompt = request.args.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Create a message to Claude
        message = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract the response content
        response = message.content[0].text
        
        return jsonify({
            "response": response
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)