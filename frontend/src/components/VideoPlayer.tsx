import React, { useRef } from "react";
import "./VideoPlayer.css";

interface VideoPlayerProps {
  src: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ src }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const handlePlay = () => {
    videoRef.current?.play();
  };

  const handlePause = () => {
    videoRef.current?.pause();
  };

  const handleStop = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  const handleTranslate = () => {
    alert("Translate functionality coming soon!");
  };

  return (
    <div className="video-player">
      <video ref={videoRef} src={src} width="640" height="360" controls={false} />
      <div className="player-controls">
        <button onClick={handlePlay}>Play</button>
        <button onClick={handlePause}>Pause</button>
        <button onClick={handleStop}>Stop</button>
        <button onClick={handleTranslate}>Translate</button>
      </div>
    </div>
  );
};

export default VideoPlayer;
