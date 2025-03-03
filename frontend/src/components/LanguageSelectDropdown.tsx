import React from "react";
import "./LanguageSelectDropdown.css"; // Importiere die angepasste CSS-Datei

interface LanguageSelectDropdownProps {
  language: "en" | "de" | "sv";
  onLanguageChange: (language: "en" | "de" | "sv") => void;
}

const LanguageSelectDropdown: React.FC<LanguageSelectDropdownProps> = ({
  language,
  onLanguageChange,
}) => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onLanguageChange(event.target.value as "en" | "de" | "sv");
  };

  return (
    <div className="language-select-container">
      <select
        className="language-dropdown"
        value={language}
        onChange={handleChange}
      >
        <option value="en">English</option>
        <option value="de">Deutsch</option>
        <option value="sv">Svenska</option>
      </select>
    </div>
  );
};

export default LanguageSelectDropdown;
