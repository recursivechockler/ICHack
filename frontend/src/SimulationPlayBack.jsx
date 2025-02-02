import React, { useState } from "react";
import "./SimulationPlayback.css";

const SimulationPlayback = ({ simulationData }) => {
  const mapData = simulationData.map;
  const gridRows = mapData.length;
  const gridCols = mapData[0].length;
  const states = simulationData.states;
  const [currentTick, setCurrentTick] = useState(0);
  const [showStats, setShowStats] = useState(false);

  const currentState = states[currentTick];

  // Build an empty grid (2D array) to place agents.
  const grid = Array.from({ length: gridRows }, () =>
    Array.from({ length: gridCols }, () => ({ attackers: [], defenders: [] }))
  );

  currentState.attackers.forEach((attacker) => {
    if (
      attacker.x >= 0 &&
      attacker.x < gridCols &&
      attacker.y >= 0 &&
      attacker.y < gridRows
    ) {
      grid[attacker.y][attacker.x].attackers.push(attacker);
    }
  });

  currentState.defenders.forEach((defender) => {
    if (
      defender.x >= 0 &&
      defender.x < gridCols &&
      defender.y >= 0 &&
      defender.y < gridRows
    ) {
      grid[defender.y][defender.x].defenders.push(defender);
    }
  });

  const renderEntity = (entity, type) => {
    // Use different emojis for alive vs dead agents.
    const baseEmoji = entity.alive ? (type === "attacker" ? "👮" : "🦹") : "💀";
    const angleDeg = (entity.orientation * 180) / Math.PI;
    return (
      <div
        key={`${type}-${entity.id}`}
        className={`entity ${type} ${entity.alive ? "" : "dead"}`}
      >
        {baseEmoji}
        {entity.alive && (
          <span
            className="viewport-arrow"
            style={{ transform: `rotate(${angleDeg}deg)` }}
          >
            ➤
          </span>
        )}
      </div>
    );
  };

  const renderCellContent = (cell, row, col) => {
    // If the underlying map cell is a wall (represented by a 1), show a wall emoji.
    if (mapData[row][col] === 1) {
      return "🟦";
    }
    return (
      <div className="cell-content">
        {cell.attackers.map((attacker) => renderEntity(attacker, "attacker"))}
        {cell.defenders.map((defender) => renderEntity(defender, "defender"))}
      </div>
    );
  };

  const renderStatsModal = () => {
    const stats = simulationData.stats;
    return (
      <div
        className="stats-modal-container"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="stats-modal">
          <h3>Simulation Statistics</h3>
          <div className="stats-content">
            {Object.entries(stats).map(([key, value]) => (
              <div key={key} className="stats-box">
                <span className="stats-key">{key}: </span>
                <span className="stats-value">{value.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <button className="tool-button" onClick={() => setShowStats(false)}>
            Close Statistics
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="toolbar">
        <h3>Simulation Playback</h3>
        <div className="slider-group">
          <label>
            Tick: {currentTick}
            <input
              type="range"
              min="0"
              max={states.length - 1}
              value={currentTick}
              onChange={(e) => setCurrentTick(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="outcome">
          Outcome:{" "}
          {simulationData.outcome.attackers_win
            ? "Attackers Win"
            : "Defenders Win"}
        </div>
        <button className="tool-button" onClick={() => setShowStats(true)}>
          Show Statistics
        </button>
      </div>

      <div className="grid">
        {grid.map((row, rowIndex) => (
          <div key={`row-${rowIndex}`} className="grid-row">
            {row.map((cell, colIndex) => (
              <div key={`cell-${rowIndex}-${colIndex}`} className="grid-cell">
                {renderCellContent(cell, rowIndex, colIndex)}
              </div>
            ))}
          </div>
        ))}
      </div>

      {showStats && (
        <div className="modal-overlay" onClick={() => setShowStats(false)}>
          {renderStatsModal()}
        </div>
      )}
    </div>
  );
};

export default SimulationPlayback;
