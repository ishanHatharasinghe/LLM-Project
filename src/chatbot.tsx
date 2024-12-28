import React, { useState } from "react";

const ChatbotForm = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse(null); // Clear previous response

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query })
      });

      if (!res.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await res.json();
      setResponse(data);
    } catch (error) {
      setResponse({ error: "Error connecting to server: " + error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Chatbot</h1>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="text"
          value={query}
          onChange={handleQueryChange}
          placeholder="Ask the chatbot..."
          required
          style={styles.input}
        />
        <button type="submit" style={styles.button}>
          Ask
        </button>
      </form>

      {loading && <p style={styles.loadingText}>Loading...</p>}

      {response && (
        <div>
          {response.error ? (
            <p style={styles.errorText}>{response.error}</p>
          ) : (
            <div>
              <h3 style={styles.responseHeader}>Response:</h3>
              <pre style={styles.response}>
                {JSON.stringify(response, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    padding: "20px",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#f9fafb",
    borderRadius: "10px",
    maxWidth: "600px",
    margin: "auto",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)"
  },
  header: {
    textAlign: "center",
    color: "#4c1d95",
    marginBottom: "20px"
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "10px"
  },
  input: {
    padding: "10px",
    fontSize: "16px",
    borderRadius: "5px",
    border: "1px solid #ddd",
    outline: "none",
    transition: "border-color 0.3s"
  },
  button: {
    padding: "10px",
    backgroundColor: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
    transition: "background-color 0.3s"
  },
  buttonHover: {
    backgroundColor: "#5b21b6"
  },
  loadingText: {
    textAlign: "center",
    color: "#2563eb",
    fontSize: "18px"
  },
  errorText: {
    color: "red",
    fontSize: "16px",
    textAlign: "center"
  },
  responseHeader: {
    fontSize: "18px",
    fontWeight: "bold",
    color: "#4c1d95"
  },
  response: {
    backgroundColor: "#f3f4f6",
    padding: "10px",
    borderRadius: "5px",
    color: "#333",
    fontSize: "14px",
    whiteSpace: "pre-wrap",
    wordWrap: "break-word"
  }
};

export default ChatbotForm;
