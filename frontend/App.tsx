import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import logo from './logo.jpeg';
import { useFocusStore } from './store';

const stateLabelMap: Record<string, string> = {
  focused: 'FOCUSED',
  distracted: 'DISTRACTED',
  fatigued: 'FATIGUED',
  stuck: 'STUCK',
  unknown: 'UNKNOWN',
};

const stateColorMap: Record<string, string> = {
  focused: '#10b981',
  distracted: '#f59e0b',
  fatigued: '#f97316',
  stuck: '#ef4444',
  unknown: '#64748b',
};

export default function App() {
  const {
    userId,
    topic,
    sessionId,
    currentState,
    messages,
    scores,
    stats,
    sessionSummary,
    dashboard,
    isLoading,
    isSessionLoading,
    error,
    setUserId,
    setTopic,
    clearError,
    startSession,
    sendMessage,
    endSession,
    uploadPdf,
  } = useFocusStore();

  const [input, setInput] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  const dailyQuotes = [
    'Pazartesi: Baslamak icin mukemmel olmana gerek yok, ama mukemmel olmak icin baslaman gerek.',
    'Sali: Bugun yapacagin kucuk bir calisma, yarinki buyuk basarin icin bir adimdir.',
    'Carsamba: Zihnini odakla, sinirlarini zorla. Basari odaklandigin yerdedir.',
    'Persembe: Zorluklar, basariyi daha tatli kilan basamaklardir. Devam et.',
    'Cuma: Haftayi guclu bitir. Disiplin, hedeflerle basari arasindaki koprudur.',
    'Cumartesi: Bugun kendine bir yatirim yap ve ogrenmeye devam et.',
    'Pazar: Yarinin kazanani, bugun vazgecmeyen kisidir.',
  ];

  const dayIndex = new Date().getDay();
  const quoteIndex = dayIndex === 0 ? 6 : dayIndex - 1;
  const todayQuote = dailyQuotes[quoteIndex];

  useEffect(() => {
    messagesRef.current?.scrollTo({
      top: messagesRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading || isSessionLoading) return;
    setInput('');
    await sendMessage(trimmed);
  };

  const handleSessionAction = async () => {
    if (isSessionLoading) return;
    if (sessionId) {
      await endSession();
    } else {
      await startSession();
    }
  };

  const handlePickPdf = () => {
    fileInputRef.current?.click();
  };

  const handlePdfSelected = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadPdf(file);
    e.target.value = '';
  };

  const currentBadgeColor = useMemo(
    () => stateColorMap[currentState] || '#64748b',
    [currentState]
  );

  const dashboardState = dashboard?.current_state ?? 'unknown';
  const dashboardBadgeColor = stateColorMap[dashboardState] || '#64748b';
  const dashboardFocusPercent =
    dashboard?.focus_score != null ? `%${Math.round(dashboard.focus_score * 100)}` : 'Yok';
  const displayedTopics =
    dashboard?.topics_covered && dashboard.topics_covered.length > 0
      ? dashboard.topics_covered
      : sessionSummary?.topics_covered ?? [];

  return (
    <div
      style={{
        backgroundColor: '#f0f4f8',
        minHeight: '100vh',
        padding: '30px',
        fontFamily: 'Inter, sans-serif',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
      }}
    >
      <header style={{ display: 'flex', alignItems: 'center', gap: '15px', flexWrap: 'wrap' }}>
        <img
          src={logo}
          alt="FocusAI Logo"
          style={{
            width: '45px',
            height: '45px',
            borderRadius: '10px',
            objectFit: 'cover',
          }}
        />
        <h1
          style={{
            color: '#1e293b',
            fontWeight: 800,
            margin: 0,
            fontSize: '24px',
          }}
        >
          FocusAI <span style={{ color: '#3b82f6' }}>Assistant</span>
        </h1>

        <div
          style={{
            backgroundColor: '#fff7ed',
            color: '#9a3412',
            border: '1px solid #fdba74',
            padding: '8px 14px',
            borderRadius: '10px',
            fontWeight: 600,
            fontSize: '13px',
            marginLeft: '10px',
          }}
        >
          Kamera analizi gecici olarak askida
        </div>
      </header>

      {error && (
        <div
          style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#991b1b',
            padding: '12px 16px',
            borderRadius: '12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <span>{error}</span>
          <button
            onClick={clearError}
            style={{
              border: 'none',
              background: 'transparent',
              color: '#991b1b',
              cursor: 'pointer',
              fontWeight: 700,
            }}
          >
            Kapat
          </button>
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 350px',
          gap: '25px',
          flex: 1,
        }}
      >
        <div
          style={{
            backgroundColor: 'white',
            borderRadius: '24px',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: '20px',
              borderBottom: '1px solid #f1f5f9',
              background: '#f8fafc',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '15px',
                flexWrap: 'wrap',
              }}
            >
              <h3 style={{ margin: 0, fontSize: '16px' }}>AI Chatbox (RAG)</h3>

              <div
                style={{
                  backgroundColor: '#eef2ff',
                  color: currentBadgeColor,
                  padding: '10px 14px',
                  borderRadius: '14px',
                  fontWeight: 700,
                  fontSize: '13px',
                  minWidth: '110px',
                  textAlign: 'center',
                }}
              >
                {stateLabelMap[currentState]}
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr auto auto',
                gap: '10px',
              }}
            >
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Kullanici ID"
                disabled={!!sessionId || isSessionLoading}
                style={{
                  padding: '12px 14px',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  outline: 'none',
                }}
              />

              <input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Bugun hangi konuyu calisacaksin?"
                disabled={!!sessionId || isSessionLoading}
                style={{
                  padding: '12px 14px',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  outline: 'none',
                }}
              />

              <button
                onClick={handleSessionAction}
                disabled={isSessionLoading || (!sessionId && !userId.trim())}
                style={{
                  backgroundColor: sessionId ? '#dc2626' : '#1e40af',
                  color: 'white',
                  border: 'none',
                  padding: '0 18px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  opacity: isSessionLoading ? 0.7 : 1,
                }}
              >
                {isSessionLoading
                  ? 'Bekleyin...'
                  : sessionId
                    ? 'Oturumu Kapat'
                    : 'Oturum Baslat'}
              </button>

              <button
                onClick={handlePickPdf}
                disabled={isLoading}
                style={{
                  backgroundColor: '#ffffff',
                  color: '#1e293b',
                  border: '1px solid #cbd5e1',
                  padding: '0 16px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                }}
              >
                PDF Yukle
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                onChange={handlePdfSelected}
                style={{ display: 'none' }}
              />
            </div>

            <div style={{ fontSize: '13px', color: '#64748b' }}>
              Session: {sessionId || 'Henuz yok'}
            </div>
          </div>

          <div
            ref={messagesRef}
            style={{
              flex: 1,
              padding: '20px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '15px',
            }}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '75%',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                }}
              >
                <div
                  style={{
                    backgroundColor: m.role === 'user' ? '#1e40af' : '#f1f5f9',
                    color: m.role === 'user' ? 'white' : '#1e293b',
                    padding: '12px 18px',
                    borderRadius: '18px',
                    fontSize: '14px',
                    whiteSpace: 'pre-wrap',
                    lineHeight: '1.5',
                  }}
                >
                  <strong>{m.role === 'ai' ? 'ai: ' : 'user: '}</strong>
                  {m.text}
                </div>

                {m.role === 'ai' && m.currentState && (
                  <div
                    style={{
                      alignSelf: 'flex-start',
                      backgroundColor: '#f8fafc',
                      color: stateColorMap[m.currentState] || '#64748b',
                      border: `1px solid ${stateColorMap[m.currentState] || '#cbd5e1'}`,
                      padding: '6px 10px',
                      borderRadius: '999px',
                      fontSize: '12px',
                      fontWeight: 700,
                    }}
                  >
                    Durum: {stateLabelMap[m.currentState]}
                  </div>
                )}

                {m.role === 'ai' && m.ragSource && (
                  <div
                    style={{
                      alignSelf: 'flex-start',
                      backgroundColor: '#ecfeff',
                      color: '#155e75',
                      border: '1px solid #a5f3fc',
                      padding: '8px 10px',
                      borderRadius: '12px',
                      fontSize: '12px',
                      lineHeight: '1.5',
                    }}
                  >
                    <strong>Not kaynagi: </strong>
                    {m.ragSource}
                  </div>
                )}

                {m.role === 'ai' && m.mentorIntervention && (
                  <div
                    style={{
                      alignSelf: 'flex-start',
                      backgroundColor: '#fff7ed',
                      color: '#9a3412',
                      border: '1px solid #fdba74',
                      padding: '10px 12px',
                      borderRadius: '14px',
                      fontSize: '13px',
                      lineHeight: '1.5',
                    }}
                  >
                    <strong>Mudahale ({m.mentorIntervention.intervention_type}): </strong>
                    {m.mentorIntervention.message}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div
            style={{
              padding: '20px',
              display: 'flex',
              gap: '10px',
              background: 'white',
              borderTop: '1px solid #f1f5f9',
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder={sessionId ? 'Mesajinizi yazin...' : 'Once oturum baslatin'}
              disabled={!sessionId || isLoading || isSessionLoading}
              style={{
                flex: 1,
                padding: '12px 20px',
                borderRadius: '12px',
                border: '1px solid #cbd5e1',
                outline: 'none',
              }}
            />
            <button
              onClick={handleSend}
              disabled={!sessionId || isLoading || isSessionLoading || !input.trim()}
              style={{
                backgroundColor: '#1e40af',
                color: 'white',
                border: 'none',
                padding: '0 25px',
                borderRadius: '12px',
                cursor: 'pointer',
                fontWeight: 'bold',
                opacity:
                  !sessionId || isLoading || isSessionLoading || !input.trim() ? 0.6 : 1,
              }}
            >
              {isLoading ? 'Gonderiliyor...' : 'Gonder'}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <div
            style={{
              backgroundColor: '#1e40af',
              padding: '20px',
              borderRadius: '24px',
              color: 'white',
              boxShadow: '0 4px 15px rgba(30, 64, 175, 0.2)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                marginBottom: '10px',
              }}
            >
              <span style={{ fontSize: '20px' }}>*</span>
              <h4 style={{ margin: 0, fontSize: '13px', letterSpacing: '0.5px' }}>
                GUNUN MOTIVASYON SOZU
              </h4>
            </div>
            <p
              style={{
                fontSize: '13.5px',
                fontStyle: 'italic',
                lineHeight: '1.5',
                margin: 0,
                opacity: 0.95,
              }}
            >
              "{todayQuote}"
            </p>
          </div>

          <div
            style={{
              backgroundColor: 'white',
              padding: '20px',
              borderRadius: '24px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
            }}
          >
            <h4
              style={{
                margin: '0 0 12px 0',
                fontSize: '13px',
                color: '#64748b',
              }}
            >
              GUNLUK VERIMLILIK
            </h4>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: '#1e293b',
                  }}
                >
                  {dashboard ? `${dashboard.message_count * 2} dk` : stats.totalTime}
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Toplam Odak</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: '#10b981',
                  }}
                >
                  {dashboard ? dashboardFocusPercent : stats.avgSuccess}
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Ort. Basari</div>
              </div>
            </div>
          </div>

          <div
            style={{
              backgroundColor: 'white',
              padding: '15px',
              borderRadius: '24px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
              height: '180px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '10px',
              }}
            >
              <h4 style={{ margin: 0, fontSize: '12px', color: '#64748b' }}>
                CANLI ODAK ANALIZI
              </h4>
              <span
                style={{
                  color: '#3b82f6',
                  fontWeight: 'bold',
                  fontSize: '12px',
                }}
              >
                %{scores.length > 0 ? scores[scores.length - 1].value : 0}
              </span>
            </div>
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scores}>
                  <YAxis domain={[0, 100]} hide />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div
            style={{
              backgroundColor: 'white',
              padding: '20px',
              borderRadius: '24px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
            }}
          >
            <h4 style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#1e293b' }}>
              Oturum Ozeti
            </h4>

            {dashboard ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div
                  style={{
                    alignSelf: 'flex-start',
                    backgroundColor: '#f8fafc',
                    color: dashboardBadgeColor,
                    border: `1px solid ${dashboardBadgeColor}`,
                    padding: '6px 10px',
                    borderRadius: '999px',
                    fontSize: '12px',
                    fontWeight: 700,
                  }}
                >
                  Son durum: {stateLabelMap[dashboardState]}
                </div>

                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Mesaj sayisi: <strong>{dashboard.message_count}</strong>
                </div>
                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Retry sayisi: <strong>{dashboard.retry_count}</strong>
                </div>
                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Mudahale sayisi: <strong>{dashboard.intervention_count}</strong>
                </div>
                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Ortalama odak: <strong>{dashboardFocusPercent}</strong>
                </div>

                {sessionSummary && (
                  <div style={{ fontSize: '14px', color: '#334155' }}>
                    Yazilan hafiza kaydi: <strong>{sessionSummary.memory_entries_written}</strong>
                  </div>
                )}

                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Islenen konular:{' '}
                  <strong>{displayedTopics.length > 0 ? displayedTopics.join(', ') : 'Yok'}</strong>
                </div>

                <div style={{ fontSize: '12px', color: '#64748b' }}>
                  Veri kaynagi: {dashboard.source || 'bilinmiyor'}
                </div>

                {dashboard.summary_text && (
                  <div
                    style={{
                      backgroundColor: '#f8fafc',
                      color: '#334155',
                      borderRadius: '14px',
                      padding: '12px',
                      fontSize: '13px',
                      lineHeight: '1.6',
                    }}
                  >
                    {/* Dashboard raporunu ayri blokta gosteriyoruz.
                        Boylece kullanici sadece sayilari degil, backend'in urettigi metin ozeti de goruyor. */}
                    {dashboard.summary_text}
                  </div>
                )}
              </div>
            ) : sessionSummary ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Yazilan hafiza kaydi: <strong>{sessionSummary.memory_entries_written}</strong>
                </div>
                <div style={{ fontSize: '14px', color: '#334155' }}>
                  Islenen konular:{' '}
                  <strong>
                    {sessionSummary.topics_covered?.length > 0
                      ? sessionSummary.topics_covered.join(', ')
                      : 'Yok'}
                  </strong>
                </div>
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>
                Dashboard verisi her mesajdan sonra ve oturum kapandiginda burada guncellenecek.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
