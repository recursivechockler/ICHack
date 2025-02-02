import React, { useState } from "react";

function GetSimulation({ getGrid, onSimulationResult }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runSimulation = async () => {
    const {
      grid,
      attacker_positions,
      defender_positions,
      attacker_params,
      defender_params,
    } = getGrid();
    setLoading(true);
    setError(null);

    const payload = {
      grid,
      attacker_positions,
      defender_positions,
      attacker_params,
      defender_params,
    };

    try {
      const response = await fetch("http://127.0.0.1:5000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      onSimulationResult(data);
      console.log(data);
    } catch (err) {
      setError(err.message);
      console.error("Error running simulation:", err);
    }
    setLoading(false);
  };

  return (
    <div>
      <button
        className="tool-button"
        onClick={runSimulation}
        disabled={loading}
      >
        {loading ? "Running Simulation..." : "Get Simulation"}
      </button>
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
    </div>
  );
}

export default GetSimulation;
