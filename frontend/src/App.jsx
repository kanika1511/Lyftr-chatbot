import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

const STARTERS = [
  "What is the capital of France?",
  "Explain LLM response caching in one paragraph.",
  "Give me three tips for resilient API design.",
];

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function newSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(16).slice(2, 10)}`;
}

function parseError(detail, status) {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((e) => {
        if (typeof e === "string") return e;
        const loc = Array.isArray(e.loc)
          ? e.loc.filter((p) => p !== "body").join(".")
          : "";
        const msg = e.msg || JSON.stringify(e);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return `HTTP ${status}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function resetSession() {
    if (loading) return;
    setSessionId(newSessionId());
    setMessages([]);
    setTimeout(() => textareaRef.current?.focus(), 0);
  }

  async function send(text) {
    const msg = typeof text === "string" ? text : input;
    if (loading) return;
    setInput("");

    const trimmed = msg.trim();
    if (trimmed) {
      setMessages((p) => [...p, { id: uid(), type: "user", content: msg }]);
    }

    setLoading(true);
    try {
      const payload = { session_id: sessionId, message: msg };
      if (systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      let body = null;
      try { body = await res.json(); } catch { /* ignore */ }

      if (!res.ok) {
        setMessages((p) => [
          ...p,
          {
            id: uid(),
            type: "error",
            content: `${res.status} — ${parseError(body?.detail, res.status)}`,
          },
        ]);
        return;
      }

      setMessages((p) => [
        ...p,
        { id: uid(), type: "assistant", content: body.reply, metadata: body.metadata },
      ]);
    } catch (err) {
      const isNetworkErr = err instanceof TypeError && err.message === "Failed to fetch";
      setMessages((p) => [
        ...p,
        {
          id: uid(),
          type: "error",
          content: isNetworkErr
            ? `Cannot reach backend at ${API_BASE_URL}. Make sure the backend is running (see README).`
            : `Network error: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app-shell">
      {/* ── Header ─────────────────────────────────── */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon" aria-hidden="true">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.2"
              strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3l7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6l7-3z"/>
              <path d="M9.5 12l2 2 3.5-3.5"/>
            </svg>
          </div>
          <div className="brand-text">
            <h1>Aegis Relay</h1>
            <p>Resilient LLM Gateway · Failover &amp; Cache</p>
          </div>
        </div>

        <div className="header-right">
          <div className="session-pill" title="Active session ID">
            <span className="live-dot" aria-hidden="true" />
            {sessionId}
          </div>
          <button className="new-btn" onClick={resetSession} disabled={loading}>
            New session
          </button>
        </div>
      </header>

      {/* ── System Prompt ────────────────────────── */}
      <section className="system-prompt-bar">
        <label htmlFor="system-prompt-input">System prompt</label>
        <input
          id="system-prompt-input"
          data-testid="system-prompt-input"
          type="text"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Optional persona or instructions for this session…"
          disabled={loading}
        />
      </section>

      {/* ── Transcript ───────────────────────────── */}
      <main className="transcript" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="1.8"
                strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <h2>Start your conversation</h2>
            <p>
              Repeated prompts are served from the LLM cache. Each reply shows
              resolution source, model, latency, and per-request token counts.
            </p>
            <div className="starter-chips">
              {STARTERS.map((s) => (
                <button key={s} className="chip" onClick={() => send(s)} disabled={loading}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-stream">
            {messages.map((m) => {
              if (m.type === "user") {
                return (
                  <div key={m.id} className="msg-row user">
                    <div className="user-bubble">{m.content}</div>
                  </div>
                );
              }

              if (m.type === "error") {
                return (
                  <div key={m.id} className="msg-row error">
                    <div className="error-card" role="alert">
                      <span className="error-label">Error</span>
                      <span>{m.content}</span>
                    </div>
                  </div>
                );
              }

              const meta = m.metadata || {};
              const isCache = meta.resolution_source === "LLM_CACHE";
              return (
                <div key={m.id} className="msg-row assistant">
                  <article className="assistant-card" data-testid="assistant-message">
                    <div className="assistant-body">{m.content}</div>
                    <div className="meta-bar">
                      <span className={`source-badge ${isCache ? "cache" : "live"}`}>
                        <span data-testid="meta-source">{meta.resolution_source}</span>
                      </span>
                      <span className="meta-sep">·</span>
                      <span className="meta-val" data-testid="meta-model">{meta.model}</span>
                      <span className="meta-sep">·</span>
                      <span className="meta-val mono" data-testid="meta-latency">{meta.latency_ms} ms</span>
                      <span className="meta-sep">·</span>
                      <span className="meta-val mono">
                        In <strong data-testid="meta-input-tokens">{meta.input_tokens}</strong>
                      </span>
                      <span className="meta-sep">·</span>
                      <span className="meta-val mono">
                        Out <strong data-testid="meta-output-tokens">{meta.output_tokens}</strong>
                      </span>
                      <span className="meta-sep">·</span>
                      <span className="meta-val mono">
                        {meta.key_id ?? "cache"}
                        {meta.fallback_attempts > 0 ? ` (+${meta.fallback_attempts})` : ""}
                      </span>
                    </div>
                  </article>
                </div>
              );
            })}

            {loading && (
              <div className="msg-row loading">
                <div className="loading-pill">
                  <span className="dots" aria-hidden="true">
                    <span /><span /><span />
                  </span>
                  Routing request…
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* ── Composer ─────────────────────────────── */}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <textarea
          ref={textareaRef}
          data-testid="chat-input"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Message Aegis Relay… (Enter to send, Shift+Enter for newline)"
          disabled={loading}
        />
        <button
          type="submit"
          className="send-btn"
          data-testid="send-button"
          disabled={loading}
        >
          {loading ? "Sending…" : "Send"}
        </button>
      </form>
    </div>
  );
}
