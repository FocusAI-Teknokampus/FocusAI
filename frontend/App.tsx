import React, { useMemo, useRef, useEffect, useState } from "react";
import { useFocusStore } from "./store";

const stateColorMap: Record<string, string> = {
  focused: "bg-green-100 text-green-700",
  distracted: "bg-yellow-100 text-yellow-700",
  fatigued: "bg-orange-100 text-orange-700",
  stuck: "bg-red-100 text-red-700",
  unknown: "bg-slate-100 text-slate-700",
};

const interventionStyleMap: Record<string, string> = {
  hint: "border-blue-200 bg-blue-50 text-blue-800",
  break: "border-green-200 bg-green-50 text-green-800",
  strategy: "border-purple-200 bg-purple-50 text-purple-800",
  question: "border-orange-200 bg-orange-50 text-orange-800",
  mode_switch: "border-indigo-200 bg-indigo-50 text-indigo-800",
  none: "border-slate-200 bg-slate-50 text-slate-800",
};

export default function App() {
  const {
    userId,
    topic,
    sessionId,
    messages,
    currentState,
    isLoading,
    error,
    sessionSummary,
    setUserId,
    setTopic,
    startSession,
    sendMessage,
    endSession,
    clearError,
  } = useFocusStore();

  const [draft, setDraft] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const canSend = !!sessionId && draft.trim().length > 0 && !isLoading;

  const handleStart = async () => {
    await startSession(userId, topic, false);
  };

  const handleSend = async () => {
    if (!canSend) return;
    const text = draft.trim();
    setDraft("");
    await sendMessage(text);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-5xl p-6">
        <div className="mb-6 rounded-3xl bg-white p-5 shadow-sm border">
          <div className="flex flex-col gap-4 md:flex-row md:items-end">
            <div className="flex-1">
              <label className="mb-1 block text-sm font-medium">Kullanıcı ID</label>
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-full rounded-xl border px-4 py-3"
                placeholder="user_001"
              />
            </div>

            <div className="flex-1">
              <label className="mb-1 block text-sm font-medium">Konu</label>
              <input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                className="w-full rounded-xl border px-4 py-3"
                placeholder="Bugün ne çalışıyorum?"
              />
            </div>

            {!sessionId ? (
              <button
                onClick={handleStart}
                disabled={isLoading || !userId.trim()}
                className="rounded-xl bg-blue-600 px-5 py-3 font-semibold text-white disabled:opacity-50"
              >
                Oturum Başlat
              </button>
            ) : (
              <button
                onClick={endSession}
                disabled={isLoading}
                className="rounded-xl bg-red-600 px-5 py-3 font-semibold text-white disabled:opacity-50"
              >
                Oturumu Kapat
              </button>
            )}
          </div>

          {sessionId && (
            <div className="mt-4 flex items-center gap-3">
              <span className="text-sm text-slate-500">Session:</span>
              <code className="rounded bg-slate-100 px-2 py-1 text-sm">{sessionId}</code>
              <span className={`rounded-full px-3 py-1 text-xs font-semibold ${stateColorMap[currentState]}`}>
                {currentState.toUpperCase()}
              </span>
            </div>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-red-700 flex items-center justify-between">
            <span>{error}</span>
            <button onClick={clearError} className="text-sm underline">Kapat</button>
          </div>
        )}

        <div className="rounded-3xl border bg-white shadow-sm">
          <div
            ref={listRef}
            className="h-[500px] overflow-y-auto p-4 flex flex-col gap-4"
          >
            {messages.length === 0 && (
              <div className="text-sm text-slate-500">
                Oturum başlatıp ilk mesajını gönder.
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className="space-y-2">
                <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "ml-auto bg-blue-600 text-white"
                    : "bg-slate-100 text-slate-900"
                }`}>
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                </div>

                {msg.role === "assistant" && msg.currentState && (
                  <div className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${stateColorMap[msg.currentState]}`}>
                    Durum: {msg.currentState.toUpperCase()}
                  </div>
                )}

                {msg.role === "assistant" && msg.mentorIntervention && (
                  <div className={`max-w-[80%] rounded-2xl border px-4 py-3 text-sm ${
                    interventionStyleMap[msg.mentorIntervention.intervention_type]
                  }`}>
                    <div className="mb-1 font-semibold">
                      Müdahale: {msg.mentorIntervention.intervention_type}
                    </div>
                    <div>{msg.mentorIntervention.message}</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="border-t p-4 flex gap-3">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={!sessionId || isLoading}
              className="flex-1 rounded-2xl border px-4 py-3"
              placeholder={sessionId ? "Mesajını yaz..." : "Önce oturum başlat"}
            />
            <button
              onClick={handleSend}
              disabled={!canSend}
              className="rounded-2xl bg-slate-900 px-5 py-3 font-semibold text-white disabled:opacity-50"
            >
              Gönder
            </button>
          </div>
        </div>

        {sessionSummary && (
          <div className="mt-6 rounded-3xl border bg-white p-5 shadow-sm">
            <h2 className="mb-3 text-lg font-bold">Oturum Özeti</h2>
            <p>Yazılan hafıza kaydı: {sessionSummary.memory_entries_written}</p>
            <p>İşlenen konular: {sessionSummary.topics_covered.join(", ") || "Yok"}</p>
          </div>
        )}
      </div>
    </div>
  );
}