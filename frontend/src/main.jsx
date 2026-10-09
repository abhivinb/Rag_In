import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function App() {
  const [conversationId, setConversationId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [query, setQuery] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_API_KEY || "");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(null);

  const headers = useMemo(() => (apiKey ? { "X-API-Key": apiKey } : {}), [apiKey]);

  async function readError(response) {
    try {
      const payload = await response.json();
      return payload.detail || `Request failed (${response.status})`;
    } catch {
      return `Request failed (${response.status})`;
    }
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!selectedFile) return;
    setUploading(true);
    setNotice(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_URL}/documents`, {
        method: "POST",
        headers,
        body: formData,
      });
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      setNotice({ type: "success", text: `${payload.file_name} indexed — ${payload.chunk_count} chunks ready.` });
      setSelectedFile(null);
      event.target.reset();
    } catch (error) {
      setNotice({ type: "error", text: error.message || "Could not reach the API." });
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion(event) {
    event?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || busy) return;
    const nextMessages = [...messages, { role: "user", content: trimmed }];
    setMessages(nextMessages);
    setQuery("");
    setBusy(true);
    setNotice(null);
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify({ query: trimmed, conversation_id: conversationId, messages }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const payload = await response.json();
      setConversationId(payload.conversation_id);
      setMessages(payload.messages || [...nextMessages, { role: "assistant", content: payload.answer }]);
      setSources(payload.sources || []);
    } catch (error) {
      setMessages(messages);
      setNotice({ type: "error", text: error.message || "Could not reach the API." });
    } finally {
      setBusy(false);
    }
  }

  function locationFor(source) {
    const metadata = source.metadata || {};
    const pages = metadata.page_numbers || (metadata.page_number ? [metadata.page_number] : []);
    if (pages.length === 1) return `Page ${pages[0]}`;
    if (pages.length > 1) return `Pages ${pages.join(", ")}`;
    return `Chunk ${(source.chunk_index || 0) + 1}`;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark">✦</div>
        <div>
          <p className="eyebrow">KNOWLEDGE WORKSPACE</p>
          <h1>Enterprise RAG</h1>
        </div>
        <div className="connection-pill"><span /> API connected at <code>{API_URL}</code></div>
      </header>

      <main className="layout">
        <section className="chat-panel panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">ASSISTANT</p>
              <h2>Ask your knowledge base</h2>
            </div>
            <button className="ghost-button" onClick={() => { setMessages([]); setSources([]); setConversationId(crypto.randomUUID()); setNotice(null); }}>
              New chat
            </button>
          </div>

          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">⌁</div>
                <h3>Start with a question</h3>
                <p>Upload a document, then ask something specific. Answers are grounded in your indexed sources.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="avatar">{message.role === "user" ? "You" : "AI"}</div>
                <div className="message-body"><span className="message-role">{message.role === "user" ? "You" : "Assistant"}</span><p>{message.content}</p></div>
              </article>
            ))}
            {busy && <div className="thinking"><span /><span /><span /> Searching your knowledge base…</div>}
          </div>

          <form className="composer" onSubmit={askQuestion}>
            <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question about your indexed documents…" maxLength={4000} rows={3} />
            <div className="composer-footer"><span>{query.length}/4000</span><button className="primary-button" disabled={!query.trim() || busy}>{busy ? "Thinking…" : "Ask assistant  ↗"}</button></div>
          </form>
        </section>

        <aside className="sidebar">
          <section className="panel upload-panel">
            <p className="eyebrow">SOURCES</p>
            <h2>Build your context</h2>
            <p className="muted">Index PDF, DOCX, or TXT files to make them available to the assistant.</p>
            <form onSubmit={uploadDocument}>
              <label className={`dropzone ${selectedFile ? "has-file" : ""}`}>
                <input type="file" accept=".pdf,.docx,.txt" onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
                <span className="upload-icon">↑</span>
                <strong>{selectedFile ? selectedFile.name : "Choose a document"}</strong>
                <small>{selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : "PDF, DOCX, or TXT · max 50 MB"}</small>
              </label>
              <button className="secondary-button" disabled={!selectedFile || uploading}>{uploading ? "Indexing…" : "Upload and index"}</button>
            </form>
          </section>

          <section className="panel settings-panel">
            <p className="eyebrow">CONNECTION</p>
            <label className="field-label" htmlFor="api-key">API key <span>(optional)</span></label>
            <input id="api-key" className="text-input" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="X-API-Key" autoComplete="off" />
            <p className="hint">Used only in this browser session.</p>
          </section>

          {notice && <div className={`notice ${notice.type}`}>{notice.type === "success" ? "✓" : "!"} {notice.text}</div>}

          {sources.length > 0 && <section className="panel sources-panel"><p className="eyebrow">LATEST ANSWER</p><h2>Source references</h2>{sources.map((source, index) => <div className="source" key={source.chunk_id || index}><div><strong>{source.metadata?.file_name || "Uploaded document"}</strong><span>{locationFor(source)}</span></div><b>{Math.round((source.retrieval_score || 0) * 100)}%</b></div>)}</section>}
        </aside>
      </main>
      <footer>Private workspace · Responses are generated from indexed documents</footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
