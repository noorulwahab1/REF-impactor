import { useState } from "react";

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<any[]>([]);

  const handleUpload = async () => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    const res = await fetch("http://localhost:8000/rank", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResults(data);
  };

  return (
    <div style={{ padding: 40 }}>
      <h1>Impact Ranking Dashboard</h1>

      <input
        type="file"
        multiple
        accept="application/pdf"
        onChange={(e) => {
          if (e.target.files) {
            setFiles(Array.from(e.target.files));
          }
        }}
      />

      <button onClick={handleUpload}>Rank Papers</button>

      {results.map((paper, index) => (
        <div key={index} style={{ marginTop: 30 }}>
          <h2>
            Rank {index + 1}: {paper.filename}
          </h2>
          <p>Predicted Score: {paper.predicted_impact_score}</p>

          <h4>Top Influential Sentences:</h4>
          {paper.top_sentences.map((s: any, i: number) => (
            <div key={i} style={{ background: "#fff3cd", padding: 10, marginTop: 5 }}>
              <strong>Importance:</strong> {s.importance_score}
              <p>{s.sentence}</p>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

export default App;