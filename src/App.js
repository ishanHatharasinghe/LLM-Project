import { Box, Button, Container, Paper, TextField, Typography } from "@mui/material";
import axios from "axios";
import React, { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleQueryChange = (e) => setQuery(e.target.value);

  const askChatbots = async () => {
    try {
      const res = await axios.post("http://localhost:8000/ask", { query });
      setResponse(res.data);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: "50px" }}>
      <Paper elevation={3} style={{ padding: "20px" }}>
        <Typography variant="h4" gutterBottom>
          Ask the Chatbot
        </Typography>
        <TextField
          label="Your Question"
          multiline
          rows={4}
          variant="outlined"
          fullWidth
          value={query}
          onChange={handleQueryChange}
        />
        <Box mt={2}>
          <Button variant="contained" color="primary" onClick={askChatbots} fullWidth>
            Ask
          </Button>
        </Box>
        {error && (
          <Typography color="error" variant="body1" style={{ marginTop: "10px" }}>
            Error: {error}
          </Typography>
        )}
        {response && (
          <Box mt={3} p={2} bgcolor="grey.100">
            <Typography variant="h6">Response:</Typography>
            <pre>{JSON.stringify(response, null, 2)}</pre>
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default App;
