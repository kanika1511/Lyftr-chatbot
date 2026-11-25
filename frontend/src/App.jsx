import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(16).slice(2, 10)}`;
}

function formatErrorDetail(detail, status) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const parts = detail.map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const loc = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== "body").join(".")
          : "";
        const msg = item.msg || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      }
      return String(item);
    });
    return parts.join("; ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return `Request failed with status ${status}.`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const transcriptEndRef = useRef(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function handleNewSession() {
    if (isLoading) return;
    setSessionId(generateSessionId());
    setMessages([]);
  }

  async function handleSend() {
    if (isLoading) return;

    const outgoingMessage = input;
    setInput("");

    const trimmed = outgoingMessage.trim();
    const userEntryId = `${Date.now()}-${Math.random().toString(16).slice(2, 6)}`;
    if (trimmed) {
      setMessages((prev) => [
        ...prev,
        { id: userEntryId, type: "user", content: outgoingMessage },
      ]);
    }

    setIsLoading(true);
    try {
      const payload = {
        session_id: sessionId,
        message: outgoingMessage,
      };
      if (systemPrompt.trim()) {
        payload.system_prompt = systemPrompt.trim();
      }

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }

      if (!response.ok) {
        const errorText = formatErrorDetail(body?.detail, response.status);
        setMessages((prev) => [
          ...prev,
          {
            id: `${Date.now()}-err`,
            type: "error",
            status: response.status,
            content: `Error (${response.status}): ${errorText}`,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-ast`,
          type: "assistant",
          content: body.reply,
          metadata: body.metadata,
        },
      ]);
    } catch (err) {
      const message =
        err instanceof Error && err.message
          ? err.message
          : "Unable to reach backend service.";
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-net`,
          type: "error",
          content: `Network error: ${message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <h1>Aegis Relay</h1>
          <p>Resilient LLM Chat Gateway with Key Failover &amp; Caching</p>
        </div>
        <div className="session-controls">
          <span className="session-pill" title="Active session ID">
            Session: {sessionId}
          </span>
          <button
            type="button"
            className="secondary-btn"
            onClick={handleNewSession}
            disabled={isLoading}
          >
            New Session
          </button>
        </div>
      </header>

      <section className="system-prompt-bar">
        <label htmlFor="system-prompt-input">System Prompt (Optional)</label>
        <input
          id="system-prompt-input"
          data-testid="system-prompt-input"
          type="text"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Leave blank to use server default system prompt..."
          disabled={isLoading}
        />
      </section>

      <main className="chat-transcript" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a multi-turn conversation</h2>
            <p>
              Each reply displays its resolution source, model, latency, and
              per-request token usage. Page loads start an isolated session.
            </p>
          </div>
        ) : (
          messages.map((item) => {
            if (item.type === "user") {
              return (
                <div key={item.id} className="message-row user">
                  <div className="bubble">{item.content}</div>
                </div>
              );
            }

            if (item.type === "error") {
              return (
                <div key={item.id} className="message-row error">
                  <div className="error-card" role="alert">
                    {item.content}
                  </div>
                </div>
              );
            }

            const meta = item.metadata || {};
            const isCache = meta.resolution_source === "LLM_CACHE";
            return (
              <div key={item.id} className="message-row assistant">
                <div className="assistant-card" data-testid="assistant-message">
                  <div className="assistant-text">{item.content}</div>
                  <div className="metadata-bar">
                    <span
                      className={`source-badge ${isCache ? "cache" : "live"}`}
                    >
                      <span data-testid="meta-source">
                        {meta.resolution_source}
                      </span>
                    </span>
                    <span className="meta-chip" data-testid="meta-model">
                      {meta.model}
                    </span>
                    <span className="meta-chip" data-testid="meta-latency">
                      {meta.latency_ms} ms
                    </span>
                    <span className="meta-chip">
                      In:{" "}
                      <span data-testid="meta-input-tokens">
                        {meta.input_tokens}
                      </span>
                    </span>
                    <span className="meta-chip">
                      Out:{" "}
                      <span data-testid="meta-output-tokens">
                        {meta.output_tokens}
                      </span>
                    </span>
                    <span className="meta-chip">
                      Key: {meta.key_id ?? "cache"}
                    </span>
                    <span className="meta-chip">
                      Retries: {meta.fallback_attempts ?? 0}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}

        {isLoading && (
          <div className="loading-indicator">Generating response...</div>
        )}
        <div ref={transcriptEndRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <textarea
          data-testid="chat-input"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-btn"
          data-testid="send-button"
          disabled={isLoading}
        >
          {isLoading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
