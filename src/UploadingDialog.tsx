import React, { useState } from "react";
import axios from "axios";

interface UploadingDialogProps {
  onClose: () => void;
}

const UploadingDialog: React.FC<UploadingDialogProps> = ({ onClose }) => {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      // First, upload the file to get its path
      const uploadRes = await axios.post(
        "http://localhost:8000/upload",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" }
        }
      );

      if (uploadRes.data.pdf_path) {
        const pdfPath = uploadRes.data.pdf_path;

        // Send the question along with the PDF path to the ask endpoint
        const questionData = { query: question, pdf_path: pdfPath };
        const askRes = await axios.post(
          "http://localhost:8000/ask",
          questionData,
          {
            headers: { "Content-Type": "application/json" }
          }
        );

        setResponse(askRes.data);
      }
    } catch (error) {
      console.error("Error uploading file or submitting question:", error);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Upload PDF and Ask Question</h2>
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        style={styles.input}
      />
      <input
        type="text"
        placeholder="Enter your question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={styles.input}
      />
      <button onClick={handleSubmit} style={styles.button}>
        Submit
      </button>
      <button onClick={onClose} style={styles.button}>
        Close
      </button>
      {response && (
        <div style={styles.response}>
          <strong>Response:</strong> {JSON.stringify(response, null, 2)}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    padding: "20px",
    backgroundColor: "#f5f5f5",
    borderRadius: "8px",
    maxWidth: "500px",
    margin: "0 auto",
    boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)"
  },
  heading: {
    color: "#4c1d95",
    textAlign: "center",
    marginBottom: "20px"
  },
  input: {
    width: "100%",
    padding: "10px",
    marginBottom: "15px",
    borderRadius: "4px",
    border: "1px solid #ddd",
    fontSize: "16px"
  },
  button: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#2563eb",
    border: "none",
    borderRadius: "4px",
    color: "#fff",
    fontSize: "16px",
    cursor: "pointer",
    marginBottom: "10px",
    transition: "background-color 0.3s"
  },
  buttonHover: {
    backgroundColor: "#5b21b6"
  },
  response: {
    marginTop: "20px",
    padding: "10px",
    backgroundColor: "#f0f0f0",
    borderRadius: "4px",
    fontSize: "14px",
    whiteSpace: "pre-wrap",
    color: "#333"
  }
};

export default UploadingDialog;
