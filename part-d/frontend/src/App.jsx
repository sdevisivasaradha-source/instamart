import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const examples = {
  policy: [
    "Who approves a refund above INR 1000?",
    "How long does a failed payment reversal take?",
    "When are delivery addresses deleted?",
  ],
  refund: [
    "Refund order ORD-100048",
    "Can I get a refund for ORD-100001?",
  ],
};

const demoProfiles = [
  {
    value: "automatic_refund",
    label: "Automatic refund · late order",
    message: "Refund order ORD-100048",
  },
  {
    value: "human_approval",
    label: "Human approval · high value",
    message: "Refund order ORD-100145",
  },
  {
    value: "on_time_denial",
    label: "Denied · delivered on time",
    message: "Refund order ORD-100001",
  },
];

async function apiRequest(path, body, token = "") {
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message =
      payload?.detail?.[0]?.msg ||
      (typeof payload?.detail === "string" ? payload.detail : null) ||
      "The request could not be completed.";
    throw new Error(message);
  }
  return response.json();
}

function SourceCard({ source }) {
  return (
    <article className="source-card">
      <div className="source-heading">
        <span>{source.source.replaceAll("_", " ")}</span>
        <strong>{Math.round(source.score * 100)}%</strong>
      </div>
      <p>{source.text}</p>
    </article>
  );
}

function App() {
  const [mode, setMode] = useState("policy");
  const [question, setQuestion] = useState(examples.policy[0]);
  const [demoProfile, setDemoProfile] = useState("automatic_refund");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => (response.ok ? response.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  function switchMode(nextMode) {
    setMode(nextMode);
    setQuestion(examples[nextMode][0]);
    setResult(null);
    setError("");
  }

  async function createDemoSession() {
    const session = await apiRequest("/api/demo/session", {
      profile: demoProfile,
    });
    setQuestion(session.suggested_message);
    return session.token;
  }

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      let payload;
      if (mode === "policy") {
        payload = await apiRequest("/api/policy/query", { question });
      } else {
        const token = await createDemoSession();
        payload = await apiRequest(
          "/api/refunds/query",
          { message: question },
          token,
        );
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="/" aria-label="Instamart Support home">
          <span className="brand-mark">i</span>
          <span>instamart <em>support</em></span>
        </a>
        <div className={`status ${health ? "online" : ""}`}>
          <span />
          {health ? `${health.mode} mode · ${health.policy_documents} policies` : "API offline"}
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow">Customer intelligence · grounded support</div>
        <h1>Answers with evidence.<br />Refunds with guardrails.</h1>
        <p>
          Ask policy questions against the approved knowledge base or run an
          authenticated refund check through the bounded support agent.
        </p>
      </section>

      <section className="workspace">
        <nav className="mode-tabs" aria-label="Copilot mode">
          <button
            className={mode === "policy" ? "active" : ""}
            disabled={loading}
            onClick={() => switchMode("policy")}
            type="button"
          >
            <span>01</span> Policy assistant
          </button>
          <button
            className={mode === "refund" ? "active" : ""}
            disabled={loading}
            onClick={() => switchMode("refund")}
            type="button"
          >
            <span>02</span> Refund agent
          </button>
        </nav>

        <div className="panel-grid">
          <form className="query-panel" onSubmit={submit}>
            <div>
              <span className="panel-label">
                {mode === "policy" ? "Grounded policy search" : "Authenticated workflow"}
              </span>
              <h2>
                {mode === "policy"
                  ? "What would you like to know?"
                  : "Which order needs review?"}
              </h2>
            </div>

            {mode === "refund" && (
              <label className="field">
                <span>Demo identity</span>
                <select
                  value={demoProfile}
                  onChange={(event) => {
                    const nextProfile = event.target.value;
                    setDemoProfile(nextProfile);
                    const profile = demoProfiles.find(
                      (candidate) => candidate.value === nextProfile,
                    );
                    setQuestion(profile.message);
                  }}
                >
                  {demoProfiles.map((profile) => (
                    <option value={profile.value} key={profile.value}>
                      {profile.label}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <label className="field">
              <span>{mode === "policy" ? "Question" : "Refund request"}</span>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows="5"
                minLength="3"
                maxLength={mode === "policy" ? 500 : 1000}
                required
              />
            </label>

            <div className="examples">
              <span>Try an example</span>
              {examples[mode].map((example) => (
                <button type="button" onClick={() => setQuestion(example)} key={example}>
                  {example}
                </button>
              ))}
            </div>

            <button className="submit" disabled={loading} type="submit">
              {loading ? "Checking…" : mode === "policy" ? "Search policies" : "Run refund check"}
              <span>→</span>
            </button>
          </form>

          <section className="response-panel" aria-live="polite">
            {!result && !error && (
              <div className="empty-state">
                <div className="orbit"><span /></div>
                <h3>Ready when you are</h3>
                <p>
                  {mode === "policy"
                    ? "Answers will include the policy source and retrieved evidence."
                    : "The agent will show each bounded tool step and its final decision."}
                </p>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {result && mode === "policy" && (
              <div className="result">
                <div className="result-meta">
                  <span className={result.grounded ? "verified" : "fallback"}>
                    {result.grounded ? "Grounded answer" : "No policy match"}
                  </span>
                  <span>{result.sources.length} source{result.sources.length === 1 ? "" : "s"}</span>
                </div>
                <h3>{result.answer}</h3>
                {result.retrieved.slice(0, 3).map((source) => (
                  <SourceCard key={`${source.source}-${source.chunk_index}`} source={source} />
                ))}
              </div>
            )}

            {result && mode === "refund" && (
              <div className="result">
                <div className="result-meta">
                  <span className={`decision ${result.status.toLowerCase()}`}>
                    {result.status.replaceAll("_", " ")}
                  </span>
                  <span>{result.tokens_used} tokens</span>
                </div>
                <h3>{result.final_answer}</h3>
                <div className="steps">
                  {result.steps.map((step, index) => (
                    <article key={`${step.tool}-${index}`}>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <div>
                        <strong>{step.tool.replaceAll("_", " ")}</strong>
                        <code>{JSON.stringify(step.observation)}</code>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <footer>
        <span>Offline teaching implementation</span>
        <span>PII-safe · source-grounded · human-supervised</span>
      </footer>
    </main>
  );
}

export default App;
