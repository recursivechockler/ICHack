import React, { useState, useEffect } from "react";
import GetSimulation from "./GetSimulation";
import "./App.css";
import SimulationPlayback from "./SimulationPlayBack";

const DEFAULT_ROWS = 20;
const DEFAULT_COLS = 20;

function createEmptyGrid(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(null));
}

function App() {
  const [selectedTool, setSelectedTool] = useState("player");
  const [gridRows, setGridRows] = useState(DEFAULT_ROWS);
  const [gridCols, setGridCols] = useState(DEFAULT_COLS);
  // We no longer have a defender slider.
  const [grid, setGrid] = useState(createEmptyGrid(DEFAULT_ROWS, DEFAULT_COLS));
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [simulationData, setSimulationData] = useState(null);
  const [showSimulation, setShowSimulation] = useState(false);

  // Simulation parameter states.
  const [attackerParams, setAttackerParams] = useState({
    vision_range: 5,
    sound_radius: 4,
    reaction: 1.0,
  });
  const [defenderParams, setDefenderParams] = useState({
    vision_range: 4,
    sound_radius: 4,
    reaction: 1.0,
  });

  // --- UPDATED processGridState ---
  // This function now checks if a cell is a wall by testing both
  // for the string "wall" or an object with type "wall".
  function processGridState() {
    const processedGrid = [];
    const attacker_positions = [];
    const defender_positions = [];
    for (let rowIndex = 0; rowIndex < grid.length; rowIndex++) {
      const newRow = [];
      for (let colIndex = 0; colIndex < grid[rowIndex].length; colIndex++) {
        const cell = grid[rowIndex][colIndex];
        if (
          cell === "wall" ||
          (cell && typeof cell === "object" && cell.type === "wall")
        ) {
          newRow.push(1);
        } else {
          newRow.push(0);
        }
        if (cell && typeof cell === "object") {
          if (cell.type === "player") {
            // Record as [x, y, orientation]
            attacker_positions.push([colIndex, rowIndex, cell.orientation]);
          } else if (cell.type === "defender") {
            defender_positions.push([colIndex, rowIndex, cell.orientation]);
          }
        }
      }
      processedGrid.push(newRow);
    }
    return {
      grid: processedGrid,
      attacker_positions,
      defender_positions,
      attacker_params: attackerParams,
      defender_params: defenderParams,
    };
  }
  // --- END UPDATED processGridState ---

  useEffect(() => {
    setGrid(createEmptyGrid(gridRows, gridCols));
  }, [gridRows, gridCols]);

  // updateCell remains the same
  const updateCell = (rowIndex, colIndex) => {
    setGrid((prevGrid) => {
      const newGrid = prevGrid.map((row) => [...row]);
      const current = newGrid[rowIndex][colIndex];
      if (selectedTool === "eraser") {
        newGrid[rowIndex][colIndex] = null;
      } else if (
        current &&
        typeof current === "object" &&
        current.type === selectedTool
      ) {
        // Rotate clockwise by 45 degrees.
        current.orientation =
          (current.orientation + Math.PI / 4) % (2 * Math.PI);
        newGrid[rowIndex][colIndex] = current;
      } else {
        // Place a new agent object with starting orientation 0.
        newGrid[rowIndex][colIndex] = { type: selectedTool, orientation: 0 };
      }
      return newGrid;
    });
  };

  const handleMouseDown = (rowIndex, colIndex) => {
    setIsMouseDown(true);
    updateCell(rowIndex, colIndex);
  };

  const handleMouseEnter = (rowIndex, colIndex) => {
    if (isMouseDown) {
      updateCell(rowIndex, colIndex);
    }
  };

  useEffect(() => {
    const handleMouseUp = () => setIsMouseDown(false);
    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, []);

  // renderCellContent remains essentially the same.
  const renderCellContent = (cellValue) => {
    if (cellValue && typeof cellValue === "object") {
      if (cellValue.type === "wall") return "🟦";
      let baseEmoji = "";
      if (cellValue.type === "player") {
        baseEmoji = "👮";
      } else if (cellValue.type === "defender") {
        baseEmoji = "🦹";
      }
      const angleDeg = (cellValue.orientation * 180) / Math.PI;
      return (
        <span>
          {baseEmoji}
          <span
            className="viewport-arrow"
            style={{ transform: `rotate(${angleDeg}deg)` }}
          >
            ➤
          </span>
        </span>
      );
    } else if (cellValue === "wall") {
      return "⬛";
    }
    return "";
  };

  return (
    <div className="app-container">
      <div className="toolbar">
        <h3>Tools</h3>
        <button
          className={`tool-button ${selectedTool === "player" ? "active" : ""}`}
          onClick={() => setSelectedTool("player")}
        >
          Player
        </button>
        <button
          className={`tool-button ${
            selectedTool === "defender" ? "active" : ""
          }`}
          onClick={() => setSelectedTool("defender")}
        >
          Defender
        </button>
        <button
          className={`tool-button ${selectedTool === "wall" ? "active" : ""}`}
          onClick={() => setSelectedTool("wall")}
        >
          Wall
        </button>
        <button
          className={`tool-button ${selectedTool === "eraser" ? "active" : ""}`}
          onClick={() => setSelectedTool("eraser")}
        >
          Eraser
        </button>

        <div className="slider-group">
          <label>
            Columns: {gridCols}
            <input
              type="range"
              min="5"
              max="40"
              value={gridCols}
              onChange={(e) => setGridCols(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Rows: {gridRows}
            <input
              type="range"
              min="5"
              max="40"
              value={gridRows}
              onChange={(e) => setGridRows(Number(e.target.value))}
            />
          </label>
        </div>
        <hr />
        <h4>Attacker Parameters</h4>
        <div className="slider-group">
          <label>
            Vision Range: {attackerParams.vision_range}
            <input
              type="range"
              min="1"
              max="20"
              step="0.5"
              value={attackerParams.vision_range}
              onChange={(e) =>
                setAttackerParams((prev) => ({
                  ...prev,
                  vision_range: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Sound Radius: {attackerParams.sound_radius}
            <input
              type="range"
              min="1"
              max="20"
              step="0.5"
              value={attackerParams.sound_radius}
              onChange={(e) =>
                setAttackerParams((prev) => ({
                  ...prev,
                  sound_radius: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Reaction: {attackerParams.reaction}
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={attackerParams.reaction}
              onChange={(e) =>
                setAttackerParams((prev) => ({
                  ...prev,
                  reaction: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
        <hr />
        <h4>Defender Parameters</h4>
        <div className="slider-group">
          <label>
            Vision Range: {defenderParams.vision_range}
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={defenderParams.vision_range}
              onChange={(e) =>
                setDefenderParams((prev) => ({
                  ...prev,
                  vision_range: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Sound Radius: {defenderParams.sound_radius}
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={defenderParams.sound_radius}
              onChange={(e) =>
                setDefenderParams((prev) => ({
                  ...prev,
                  sound_radius: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
        <div className="slider-group">
          <label>
            Reaction: {defenderParams.reaction}
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={defenderParams.reaction}
              onChange={(e) =>
                setDefenderParams((prev) => ({
                  ...prev,
                  reaction: Number(e.target.value),
                }))
              }
            />
          </label>
        </div>
      </div>

      <div
        className="grid"
        onMouseUp={() => setIsMouseDown(false)}
        onMouseLeave={() => setIsMouseDown(false)}
      >
        {grid.map((row, rowIndex) => (
          <div key={rowIndex} className="grid-row">
            {row.map((cell, colIndex) => (
              <div
                key={colIndex}
                className="grid-cell"
                onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
              >
                {renderCellContent(cell)}
              </div>
            ))}
          </div>
        ))}
      </div>

      <GetSimulation
        getGrid={processGridState}
        onSimulationResult={(result) => {
          setSimulationData(result);
          setShowSimulation(true);
        }}
      />

      {simulationData && showSimulation && (
        <div className="modal-overlay" onClick={() => setShowSimulation(false)}>
          <div
            className="simulation-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <SimulationPlayback simulationData={simulationData} />
            <button
              className="tool-button"
              onClick={() => setShowSimulation(false)}
            >
              Close Simulation
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
