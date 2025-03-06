interface TabNavigationProps {
  setActivePage: React.Dispatch<React.SetStateAction<"live" | "video">>;
}

const TabNavigation: React.FC<TabNavigationProps> = ({ setActivePage }) => {
  const buttons = ["LIVE", "VIDEO", "Detection", "Camera", "Upload File"];

  return (
    <div className="nav-controls">
      {buttons.map((button) => (
        <button
          key={button}
          onClick={() => {
            if (button === "LIVE") setActivePage("live");
            if (button === "VIDEO") setActivePage("video");
            if (button === "Detection") console.log("Detection selected");
            if (button === "Camera") console.log("Camera selected");
            if (button === "Upload File") console.log("Upload File selected");
          }}
        >
          {button}
        </button>
      ))}
    </div>
  );
};

export default TabNavigation;
