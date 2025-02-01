import React, { useState } from "react";
import "./SimulationPlayback.css";

const SimulationPlayback = ({ simulationData }) => {
  const mapData = simulationData.map;
  const gridRows = mapData.length;
  const gridCols = mapData[0].length;
  const states = simulationData.states;
  const [currentTick, setCurrentTick] = useState(0);

  const currentState = states[currentTick];

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
    const baseEmoji = entity.alive ? (type === "attacker" ? "🙂" : "😎") : "💀";
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
    if (mapData[row][col] === 1) {
      return "⬛";
    }
    return (
      <div className="cell-content">
        {cell.attackers.map((attacker) => renderEntity(attacker, "attacker"))}
        {cell.defenders.map((defender) => renderEntity(defender, "defender"))}
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
    </div>
  );
};

export default SimulationPlayback;
