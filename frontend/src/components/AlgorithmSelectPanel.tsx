import { useState } from "react";
import { ModelName } from "../types/custom_types";
import "./AlgorithmSelectPanel.css";

interface PanelProps {
  onButtonClick: (name: ModelName) => void;
}

function AlgorithmSelectPanel({ onButtonClick }: PanelProps) {
  const [isNoneButtonActive, setIsNoneButtonActive] = useState(true);

  const handleButtonClick = (button: string) => {
    if (button === "None") {
      setIsNoneButtonActive(true);
      onButtonClick(ModelName.NONE);
    }
    if (button === "ASL Fingerspelling") {
      setIsNoneButtonActive(false);
      onButtonClick(ModelName.ASLFINGERSPELLING);
    }
  };

  return (
    <div className="buttons">
      <button
        className={isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("None")}
      >
        None
      </button>
      <button
        className={!isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("ASL Fingerspelling")}
      >
        ASL Fingerspelling
      </button>
    </div>
  );
}
export default AlgorithmSelectPanel;
