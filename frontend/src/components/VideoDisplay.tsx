import "./VideoDisplay.css";

interface Props {
  streamUrl: string;
}

function VideoDisplay({ streamUrl }: Props) {
  return (
    <div>
      <img src={streamUrl} alt="No input available..." />
    </div>
  );
}

export default VideoDisplay;
