import { useState, useRef, useEffect } from "react";
import { useAuth } from "../../context/AuthContext";

const BACKEND = "https://cognisafe-backend-yh4x.onrender.com";

const fetchLatestSession = async (token) => {
  try {
    const res = await fetch(`${BACKEND}/api/sessions/latest`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
};

const buildSystemPrompt = (user, session) => {
  let prompt = `You are CogniSafe's friendly AI health assistant. CogniSafe is a voice biomarker platform that detects early cognitive health changes by analysing 14 biomarkers from daily 3-minute voice recordings.

You help users understand:
- Their biomarker results (semantic coherence, lexical diversity, speech rate, pause frequency, HNR, jitter, shimmer, pitch mean/range, articulation rate, idea density, syntactic complexity, pause duration, filled pause rate)
- What risk tiers mean (Green = good, Yellow = watch, Orange = alert, Red = concern)
- General cognitive and brain health questions
- How the CogniSafe pipeline works
- What anomaly flags mean and what to do about them

Always be warm, reassuring, and clear. Never diagnose. Always recommend consulting a doctor for medical concerns. Keep responses concise — 2-4 sentences unless the user asks for more detail.

User's name: ${user?.name || "there"}`;

  if (session) {
    const bm = session.biomarkers || session;
    prompt += `

Their most recent session results:
- Risk Tier: ${session.risk_tier || "Unknown"}
- Semantic Coherence: ${bm.semantic_coherence?.toFixed(3) ?? "N/A"}
- Lexical Diversity: ${bm.lexical_diversity?.toFixed(2) ?? "N/A"}
- Speech Rate: ${bm.speech_rate ? Math.round(bm.speech_rate) + " wpm" : "N/A"}
- Pause Frequency: ${bm.pause_frequency?.toFixed(1) ?? "N/A"}/min
- HNR: ${bm.hnr?.toFixed(1) ?? "N/A"} dB
- Jitter: ${bm.jitter ? (bm.jitter * 100).toFixed(2) + "%" : "N/A"}
- Shimmer: ${bm.shimmer ? (bm.shimmer * 100).toFixed(1) + "%" : "N/A"}
- Idea Density: ${bm.idea_density?.toFixed(3) ?? "N/A"}
- Syntactic Complexity: ${bm.syntactic_complexity?.toFixed(2) ?? "N/A"}

When the user asks about their results, refer to these specific values.`;
  } else {
    prompt += `\n\nThe user hasn't completed a session yet. Encourage them to record their first session.`;
  }

  return prompt;
};

const BrainIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2a2.5 2.5 0 0 1 5 0v.5" />
    <path d="M9 2.5C6.5 3 4.5 5.5 4.5 8.5c0 1.5.5 2.8 1.3 3.8" />
    <path d="M15 2.5c2.5.5 4.5 3 4.5 6 0 1.5-.5 2.8-1.3 3.8" />
    <path d="M5.8 12.3A4.5 4.5 0 0 0 4.5 15c0 2.5 2 4.5 4.5 4.5h6c2.5 0 4.5-2 4.5-4.5a4.5 4.5 0 0 0-1.3-3.1" />
    <path d="M9 15v2M12 14v3M15 15v2" />
  </svg>
);

const CloseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const ChatBot = () => {
  const { user, token, isLoggedIn } = useAuth();
  const [open, setOpen]               = useState(false);
  const [messages, setMessages]       = useState([]);
  const [input, setInput]             = useState("");
  const [loading, setLoading]         = useState(false);
  const [sessionData, setSessionData] = useState(null);
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    if (isLoggedIn && token && !sessionLoaded) {
      fetchLatestSession(token).then((s) => {
        setSessionData(s);
        setSessionLoaded(true);
      });
    }
  }, [isLoggedIn, token, sessionLoaded]);

  useEffect(() => {
    if (open && messages.length === 0) {
      const name = user?.name?.split(" ")[0] || "there";
      const greeting = sessionData
        ? `Hi ${name}! 👋 I'm your CogniSafe health assistant. I can see your latest session results — ask me anything about your biomarkers, risk tier, or general brain health!`
        : `Hi ${name}! 👋 I'm your CogniSafe health assistant. Ask me anything about how CogniSafe works, what the biomarkers mean, or general cognitive health questions!`;
      setMessages([{ role: "assistant", content: greeting }]);
    }
    if (open) setTimeout(() => inputRef.current?.focus(), 100);
  }, [open]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (overrideInput) => {
    const text = (overrideInput ?? input).trim();
    if (!text || loading) return;

    setInput("");
    const newMessages = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          system_prompt: buildSystemPrompt(user, sessionData),
          messages: newMessages,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Chat error");
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I'm having trouble connecting right now. Please try again in a moment." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!isLoggedIn) return null;

  return (
    <>
      <style>{`
        .cog-chat-bubble {
          position: fixed; bottom: 28px; right: 28px; z-index: 9999;
          width: 52px; height: 52px; border-radius: 50%;
          background: linear-gradient(135deg, #C4B0F8, #9B4F7A);
          border: none; cursor: pointer;
          display: flex; align-items: center; justify-content: center; color: #fff;
          box-shadow: 0 4px 20px rgba(155,79,122,0.45);
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .cog-chat-bubble:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(155,79,122,0.55); }
        .cog-chat-bubble.open { background: linear-gradient(135deg, #9B4F7A, #7B3560); }
        .cog-chat-popup {
          position: fixed; bottom: 92px; right: 28px; z-index: 9998;
          width: 360px; height: 500px;
          background: var(--bg-raised, #fff);
          border-radius: 20px;
          border: 1px solid var(--rose-border, rgba(155,79,122,0.18));
          box-shadow: 0 16px 60px rgba(46,21,37,0.18);
          display: flex; flex-direction: column; overflow: hidden;
          animation: chatPopIn 0.25s cubic-bezier(0.34,1.56,0.64,1) both;
        }
        @keyframes chatPopIn {
          from { opacity: 0; transform: scale(0.88) translateY(16px); transform-origin: bottom right; }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
        .cog-chat-header {
          padding: 16px 18px;
          background: linear-gradient(135deg, #C4B0F8 0%, #9B4F7A 100%);
          display: flex; align-items: center; gap: 10px; flex-shrink: 0;
        }
        .cog-chat-header-icon {
          width: 36px; height: 36px; background: rgba(255,255,255,0.2);
          border-radius: 50%; display: flex; align-items: center; justify-content: center;
          color: #fff; flex-shrink: 0;
        }
        .cog-chat-header-text { flex: 1; }
        .cog-chat-header-title {
          font-family: 'Playfair Display', serif; font-size: 15px;
          font-weight: 600; color: #fff; line-height: 1.2;
        }
        .cog-chat-header-sub { font-size: 11px; color: rgba(255,255,255,0.75); margin-top: 1px; }
        .cog-chat-close {
          background: rgba(255,255,255,0.18); border: none; border-radius: 8px;
          width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
          cursor: pointer; color: #fff; transition: background 0.15s; flex-shrink: 0;
        }
        .cog-chat-close:hover { background: rgba(255,255,255,0.3); }
        .cog-chat-messages {
          flex: 1; overflow-y: auto; padding: 14px 14px 8px;
          display: flex; flex-direction: column; gap: 10px;
          scrollbar-width: thin;
          scrollbar-color: var(--rose-border, rgba(155,79,122,0.2)) transparent;
        }
        .cog-chat-messages::-webkit-scrollbar { width: 4px; }
        .cog-chat-messages::-webkit-scrollbar-thumb { background: var(--rose-border, rgba(155,79,122,0.2)); border-radius: 4px; }
        .cog-msg {
          max-width: 85%; padding: 9px 13px; border-radius: 14px;
          font-size: 13.5px; line-height: 1.5; word-break: break-word;
        }
        .cog-msg.user {
          align-self: flex-end;
          background: linear-gradient(135deg, #C4B0F8, #9B4F7A);
          color: #fff; border-bottom-right-radius: 4px;
        }
        .cog-msg.assistant {
          align-self: flex-start;
          background: var(--bg-surface, #F3EDF3);
          color: var(--text, #140A10);
          border-bottom-left-radius: 4px;
          border: 1px solid var(--rose-border, rgba(155,79,122,0.12));
        }
        .cog-chat-typing {
          align-self: flex-start;
          background: var(--bg-surface, #F3EDF3);
          border: 1px solid var(--rose-border, rgba(155,79,122,0.12));
          padding: 10px 14px; border-radius: 14px; border-bottom-left-radius: 4px;
          display: flex; gap: 4px; align-items: center;
        }
        .cog-dot {
          width: 6px; height: 6px; background: var(--rose, #9B4F7A);
          border-radius: 50%; animation: cogDot 1.2s infinite ease-in-out; opacity: 0.5;
        }
        .cog-dot:nth-child(2) { animation-delay: 0.2s; }
        .cog-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes cogDot {
          0%,80%,100% { transform: scale(1); opacity: 0.5; }
          40% { transform: scale(1.3); opacity: 1; }
        }
        .cog-chat-suggestions {
          display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 10px;
        }
        .cog-suggestion {
          font-size: 11.5px; padding: 5px 10px; border-radius: 20px;
          border: 1px solid var(--rose-border, rgba(155,79,122,0.2));
          background: var(--rose-pale, rgba(155,79,122,0.05));
          color: var(--rose, #9B4F7A); cursor: pointer;
          font-family: 'Inter', sans-serif; transition: background 0.15s; white-space: nowrap;
        }
        .cog-suggestion:hover { background: rgba(155,79,122,0.12); }
        .cog-chat-input-row {
          padding: 10px 12px;
          border-top: 1px solid var(--rose-border, rgba(155,79,122,0.12));
          display: flex; gap: 8px; align-items: flex-end;
          flex-shrink: 0; background: var(--bg-raised, #fff);
        }
        .cog-chat-input {
          flex: 1; border: 1px solid var(--rose-border, rgba(155,79,122,0.2));
          border-radius: 12px; padding: 9px 13px; font-size: 13.5px;
          font-family: 'Inter', sans-serif;
          background: var(--bg-surface, #F3EDF3); color: var(--text, #140A10);
          resize: none; outline: none; max-height: 90px; line-height: 1.4;
          transition: border-color 0.15s;
        }
        .cog-chat-input:focus { border-color: var(--rose, #9B4F7A); }
        .cog-chat-input::placeholder { color: var(--text-muted, #A08898); }
        .cog-chat-send {
          width: 36px; height: 36px;
          background: linear-gradient(135deg, #C4B0F8, #9B4F7A);
          border: none; border-radius: 10px; cursor: pointer;
          display: flex; align-items: center; justify-content: center; color: #fff;
          flex-shrink: 0; transition: opacity 0.15s, transform 0.15s;
        }
        .cog-chat-send:hover:not(:disabled) { opacity: 0.88; transform: scale(1.05); }
        .cog-chat-send:disabled { opacity: 0.4; cursor: not-allowed; }
        @media (max-width: 480px) {
          .cog-chat-popup { width: calc(100vw - 24px); right: 12px; bottom: 80px; height: 460px; }
          .cog-chat-bubble { bottom: 20px; right: 16px; }
        }
      `}</style>

      <button
        className={`cog-chat-bubble ${open ? "open" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-label="Open CogniSafe assistant"
      >
        {open ? <CloseIcon /> : <BrainIcon />}
      </button>

      {open && (
        <div className="cog-chat-popup">
          <div className="cog-chat-header">
            <div className="cog-chat-header-icon"><BrainIcon /></div>
            <div className="cog-chat-header-text">
              <div className="cog-chat-header-title">CogniSafe Assistant</div>
              <div className="cog-chat-header-sub">
                {sessionData ? `Last session: ${sessionData.risk_tier || "Green"}` : "Ask me anything"}
              </div>
            </div>
            <button className="cog-chat-close" onClick={() => setOpen(false)}><CloseIcon /></button>
          </div>

          <div className="cog-chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`cog-msg ${msg.role}`}>{msg.content}</div>
            ))}
            {loading && (
              <div className="cog-chat-typing">
                <div className="cog-dot" /><div className="cog-dot" /><div className="cog-dot" />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {messages.length <= 1 && !loading && (
            <div className="cog-chat-suggestions">
              {["What does my risk tier mean?", "Explain semantic coherence", "Is my speech rate normal?", "What is jitter?"].map((s) => (
                <button key={s} className="cog-suggestion" onClick={() => sendMessage(s)}>{s}</button>
              ))}
            </div>
          )}

          <div className="cog-chat-input-row">
            <textarea
              ref={inputRef}
              className="cog-chat-input"
              placeholder="Ask about your results..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
            />
            <button
              className="cog-chat-send"
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
            >
              <SendIcon />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;