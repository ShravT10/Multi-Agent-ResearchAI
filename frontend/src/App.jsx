import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Search, Loader2, CheckCircle2, AlertCircle, Cpu, BookOpen } from 'lucide-react';

function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleResearch = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/research', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch research results');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Research Assistant</h1>
        <p className="subtitle">AI-powered multi-agent deep research system</p>
      </header>

      <form onSubmit={handleResearch} className="search-box">
        <Cpu className="fact-icon" style={{ marginLeft: '10px', color: '#8b949e' }} size={20} />
        <input
          type="text"
          placeholder="Enter your research question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? <Loader2 className="spinner-icon animate-spin" size={20} /> : <Search size={20} />}
          {loading ? 'Analyzing...' : 'Research'}
        </button>
      </form>

      {error && (
        <div className="report-container" style={{ borderColor: '#f85149', background: 'rgba(248, 81, 73, 0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#f85149' }}>
            <AlertCircle size={20} />
            <p>{error}</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Deploying agents to gather and analyze data...</p>
        </div>
      )}

      {result && result.report && (
        <main className="report-container">
          <div className="markdown-content">
            <ReactMarkdown>{result.report}</ReactMarkdown>
          </div>

          {result.analysis && result.analysis.per_task_analysis && (
            <section className="analysis-section">
              <h2><BookOpen size={24} style={{ verticalAlign: 'middle', marginRight: '10px' }} /> Deep Analysis</h2>
              {Object.entries(result.analysis.per_task_analysis).map(([taskId, analysis]) => (
                <div key={taskId} className="task-analysis-card">
                  <span className="task-title">Component Analysis #{taskId}</span>
                  <ul className="facts-list">
                    {analysis.verified_facts.map((fact, index) => (
                      <li key={index} className="fact-item">
                        <CheckCircle2 size={16} className="fact-icon" />
                        <span>{fact}</span>
                      </li>
                    ))}
                    {analysis.key_insights.map((insight, index) => (
                      <li key={index} className="fact-item" style={{ color: '#8b949e' }}>
                        <BookOpen size={16} style={{ marginTop: '4px' }} />
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </section>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
