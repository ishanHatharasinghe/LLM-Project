import React from "react";
import Chatbot from "./Chatbot";
import UploadingDialog from "./UploadingDialog";

const App: React.FC = () => {
  return (
    <div className="App">
      <Chatbot />
    </div>
  );
};

const App: React.FC = () => {
  const [showDialog, setShowDialog] = useState(false);

  const handleDialogClose = () => {
    setShowDialog(false);
  };

  return (
    <div className="App">
      <h1>PDF Question-Answer System</h1>
      <button onClick={() => setShowDialog(true)}>Upload PDF</button>
      {showDialog && <UploadingDialog onClose={handleDialogClose} />}
    </div>
  );
};

export default App;
