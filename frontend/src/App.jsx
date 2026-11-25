import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

const STARTER_PROMPTS = [
  "What is the capital of France?",
  "Summarize why response caching matters in 2 sentences.",
  "Give me three concise tips for resilient APIs.",
];

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
  const inputRef = useRef(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function handleNewSession() {
    if (isLoading) return;
    setSessionId(generateSessionId());
    setMessages([]);
    inputRef.current?.focus();
  }

  async function sendMessage(rawMessage) {
    if (isLoading) return;

    const outgoingMessage = rawMessage;
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
            content: `Request failed (${response.status}): ${errorText}`,
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

  function handleSend() {
    return sendMessage(input);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="page-backdrop">
      <div className="workspace-shell">
        <header className="topbar">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true">
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.1"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 3l7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6l7-3z" />
                <path d="M9.5 12l2 2 3.5-3.5" />
              </svg>
            </div>
            <div className="brand-copy">
              <h1>Aegis Relay</h1>
              <span>Failover &amp; Cached Gateway</span>
            </div>
          </div>

          <div className="session-bar">
            <div className="session-tag" title="Active session ID">
              <span className="status-dot" aria-hidden="true" />
              <span>{sessionId}</span>
            </div>
            <button
              type="button"
              className="new-session-btn"
              onClick={handleNewSession}
              disabled={isLoading}
            >
              New session
            </button>
          </div>
        </header>

        <section className="prompt-strip">
          <label htmlFor="system-prompt-input">System prompt</label>
          <input
            id="system-prompt-input"
            data-testid="system-prompt-input"
            type="text"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional persona or instruction for this session..."
            disabled={isLoading}
          />
        </section>

        <main className="transcript" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-canvas">
              <div className="empty-emblem" aria-hidden="true">
                <svg
                  width="22"
                  height="22"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <h2>Ready for your first prompt</h2>
              <p>
                Conversations persist per session. Repeated prompts serve cleanly
                from cache with lower latency and per-turn token counts.
              </p>
              <div className="starter-row">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="starter-chip"
                    onClick={() => sendMessage(prompt)}
                    disabled={isLoading}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-stack">
              {messages.map((item) => {
                if (item.type === "user") {
                  return (
                    <div key={item.id} className="turn-row turn-user">
                      <div className="user-bubble">{item.content}</div>
                    </div>
                  );
                }

                if (item.type === "error") {
                  return (
                    <div key={item.id} className="turn-row turn-error">
                      <div className="error-banner" role="alert">
                        <span className="error-label">Error</span>
                        <span>{item.content}</span>
                      </div>
                    </div>
                  );
                }

                const meta = item.metadata || {};
                const isCache = meta.resolution_source === "LLM_CACHE";
                return (
                  <div key={item.id} className="turn-row turn-assistant">
                    <article
                      className="assistant-card"
                      data-testid="assistant-message"
                    >
                      <div className="assistant-prose">{item.content}</div>
                      <footer className="telemetry-row">
                        <span
                          className={`source-tag ${isCache ? "is-cache" : "is-live"}`}
                        >
                          <span data-testid="meta-source">
                            {meta.resolution_source}
                          </span>
                        </span>
                        <span className="telemetry-sep" aria-hidden="true">
                          ·
                        </span>
                        <span
                          className="telemetry-value"
                          data-testid="meta-model"
                        >
                          {meta.model}
                        </span>
                        <span className="telemetry-sep" aria-hidden="true">
                          ·
                        </span>
                        <span
                          className="telemetry-value mono"
                          data-testid="meta-latency"
                        >
                          {meta.latency_ms} ms
                        </span>
                        <span className="telemetry-sep" aria-hidden="true">
                          ·
                        </span>
                        <span className="telemetry-value mono">
                          In{" "}
                          <strong data-testid="meta-input-tokens">
                            {meta.input_tokens}
                          </strong>
                        </span>
                        <span className="telemetry-sep" aria-hidden="true">
                          ·
                        </span>
                        <span className="telemetry-value mono">
                          Out{" "}
                          <strong data-testid="meta-output-tokens">
                            {meta.output_tokens}
                          </strong>
                        </span>
                        <span className="telemetry-sep" aria-hidden="true">
                          ·
                        </span>
                        <span className="telemetry-value mono">
                          {meta.key_id ?? "cache"}
                          {meta.fallback_attempts > 0
                            ? ` (${meta.fallback_attempts} retry)`
                            : ""}
                        </span>
                      </footer>
                    </article>
                  </div>
                );
              })}

              {isLoading && (
                <div className="turn-row turn-assistant">
                  <div className="pending-pill" aria-label="Generating response">
                    <span className="pulse-dots">
                      <span />
                      <span />
                      <span />
                    </span>
                    <span>Routing request…</span>
                  </div>
                </div>
              )}
              <div ref={transcriptEndRef} />
            </div>
          )}
        </main>

        <form
          className="composer-dock"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Aegis Relay..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="submit-btn"
            data-testid="send-button"
            disabled={isLoading}
          >
            {isLoading ? "Sending" : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
