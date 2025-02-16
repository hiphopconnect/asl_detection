import "./VideoDisplay.css";

interface Props {
  streamUrl: string;
}

function VideoDisplay(props: Props) {
  return (
    <div>
      <img src={props.streamUrl} alt="No input available..."></img>
    </div>
  );
}

export default VideoDisplay;
