import React from "react";
import "./VideoPlayer.css";

interface GifPlayerProps {
  src: string;
}

const GifPlayer: React.FC<GifPlayerProps> = ({ src }) => {
  return (
    <div className="gif-player">
      <div className="gif-container">
        <img
          src={src}
          alt="Animation"
          style={{ maxWidth: "100%", maxHeight: "80vh" }}
        />
        <p className="gif-notice"></p>
      </div>
    </div>
  );
};

export default GifPlayer;
