import { useState } from "react";
import { ModelName } from "../types/custom_types";
import "./AlgorithmSelectPanel.css";

const alphabetImages = [
  "/images/a.jpg",
  "/images/b.jpg",
  "/images/c.jpg",
  "/images/d.jpg",
  "/images/e.jpg",
  "/images/f.jpg",
  "/images/g.jpg",
  "/images/h.jpg",
  "/images/i.jpg",
  "/images/k.jpg",
  "/images/l.jpg",
  "/images/m.jpg",
  "/images/n.jpg",
  "/images/o.jpg",
  "/images/p.jpg",
  "/images/q.jpg",
  "/images/r.jpg",
  "/images/s.jpg",
  "/images/t.jpg",
  "/images/u.jpg",
  "/images/v.jpg",
  "/images/w.jpg",
  "/images/x.jpg",
  "/images/y.jpg",
];

interface PanelProps {
  onButtonClick: (name: ModelName) => void;
  language: "en" | "de" | "sv";
}

const translations = {
  en: { none: "None" },
  de: { none: "Keins" },
  sv: { none: "Ingen" },
};

function AlgorithmSelectPanel({ onButtonClick, language }: PanelProps) {
  const [isNoneButtonActive, setIsNoneButtonActive] = useState(true);
  const [isContainerVisible, setIsContainerVisible] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  // Handle Button Click for "None", "ASL Fingerspelling", "ASL Word Detection" and "ABC"
  const handleButtonClick = (button: string) => {
    if (button === "None") {
      setIsNoneButtonActive(true);
      onButtonClick(ModelName.NONE);
      setIsContainerVisible(false); // Container verstecken
    }
    if (button === "ASL Fingerspelling") {
      setIsNoneButtonActive(false);
      onButtonClick(ModelName.ASLFINGERSPELLING);
      setIsContainerVisible(false); // hide Container
    }
    if (button === "ASL Word Detection") {
      setIsNoneButtonActive(false);
      onButtonClick(ModelName.ASLWORDDETECTION);
      setIsContainerVisible(false); // hide Container
    }
    if (button === "ABC") {
      setIsContainerVisible((prev) => !prev); // en-/disable Container
    }
  };

  return (
    <div className="buttons">
      <button
        className={isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("None")}
      >
        {translations[language].none}
      </button>
      <button
        className={!isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("ASL Fingerspelling")}
      >
        ASL Fingerspelling
      </button>
      <button
        className={!isNoneButtonActive ? "active" : ""}
        onClick={() => handleButtonClick("ASL Word Detection")}
      >
        ASL Word Detection
      </button>

      {/* ABC Button only showes if algorithm is selected */}
      {!isNoneButtonActive && (
        <div className="mini-button-container">
          <button
            className="mini-button"
            onClick={() => handleButtonClick("ABC")}
          >
            {"ABC"}
          </button>
        </div>
      )}

      {/* Container for Pictures */}
      {isContainerVisible && (
        <div className="image-container">
          <button
            className="navigate-button"
            onClick={() =>
              setCurrentImageIndex((prev) =>
                prev === 0 ? alphabetImages.length - 1 : prev - 1
              )
            }
          >
            &lt;
          </button>

          {alphabetImages[currentImageIndex] ? (
            <img
              src={alphabetImages[currentImageIndex] || ""}
              alt={`Alphabet letter ${String.fromCharCode(
                65 + currentImageIndex
              )}`}
              className="image-display"
            />
          ) : (
            <p>No image available</p>
          )}

          <button
            className="navigate-button"
            onClick={() =>
              setCurrentImageIndex((prev) =>
                prev === alphabetImages.length - 1 ? 0 : prev + 1
              )
            }
          >
            &gt;
          </button>
        </div>
      )}
    </div>
  );
}

export default AlgorithmSelectPanel;
