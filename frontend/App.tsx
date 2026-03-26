import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts';
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

const cardStyle: React.CSSProperties = {
  backgroundColor: 'white',
  padding: '18px',
  borderRadius: '24px',
  boxShadow: '0 14px 32px rgba(15, 23, 42, 0.06)',
};

function formatPercent(value?: number | null) {
  if (value == null) return 'Yok';
  return `%${Math.round(value * 100)}`;
}

function formatDateTime(value?: string | null) {
  if (!value) return 'Yok';
  return new Date(value).toLocaleString();
}

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
    uploadedDocuments,
    sessionHistory,
    selectedHistorySessionId,
    selectedHistoryMessages,
    focusTrend,
    welcomeData,
    feedbackNotice,
    isLoading,
    isSessionLoading,
    error,
    setUserId,
    setTopic,
    clearError,
    clearFeedbackNotice,
    startSession,
    sendMessage,
    endSession,
    uploadPdf,
    hydrateUserWorkspace,
    selectHistorySession,
    submitFeedback,
  } = useFocusStore();

  const [input, setInput] = useState('');
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 1180);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  const dailyQuotes = [
    'Pazartesi: Baslamak icin mukemmel olmana gerek yok.',
    'Sali: Kucuk bir calisma, yarinki buyuk basarin icin adimdir.',
    'Carsamba: Basari odaklandigin yerdedir.',
    'Persembe: Zorluklar sadece basamaklardir.',
    'Cuma: Disiplin, hedeflerle basari arasindaki koprudur.',
    'Cumartesi: Bugun kendine bir yatirim yap.',
    'Pazar: Yarinin kazanani, bugun vazgecmeyendir.',
  ];

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 1180);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    messagesRef.current?.scrollTo({
      top: messagesRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

  useEffect(() => {
    const normalizedUserId = userId.trim();
    if (!normalizedUserId) return;
    void hydrateUserWorkspace(normalizedUserId);
  }, [userId, hydrateUserWorkspace]);

  const currentBadgeColor = useMemo(
    () => stateColorMap[currentState] || '#64748b',
    [currentState]
  );

  const quoteIndex = new Date().getDay() === 0 ? 6 : new Date().getDay() - 1;
  const selectedSession = sessionHistory.find(
    (item) => item.session_id === selectedHistorySessionId
  );
  const feedbackSessionId =
    sessionId || welcomeData?.last_session?.session_id || selectedHistorySessionId;
  const focusTrendChartData =
    focusTrend?.points.map((point) => ({
      label: point.date.slice(5),
      value: Math.round(point.focus_score * 100),
    })) ?? [];

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading || isSessionLoading) return;
    setInput('');
    await sendMessage(trimmed);
  };

  const handleSessionAction = async () => {
    if (sessionId) {
      await endSession();
      return;
    }
    await startSession();
  };

  const handlePdfSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadPdf(file);
    e.target.value = '';
  };

  const renderFeedbackButtons = (interventionType?: string | null) => {
    if (!feedbackSessionId || !interventionType) return null;

    const pillStyle: React.CSSProperties = {
      border: '1px solid #fb923c',
      backgroundColor: 'white',
      color: '#9a3412',
      borderRadius: '999px',
      padding: '6px 10px',
      fontSize: '12px',
      fontWeight: 700,
      cursor: 'pointer',
    };

    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        <button
          onClick={() =>
            submitFeedback({
              feedbackType: 'correct_detection',
              interventionType,
              sessionId: feedbackSessionId,
            })
          }
          style={pillStyle}
        >
          Dogru tespit
        </button>
        <button
          onClick={() =>
            submitFeedback({
              feedbackType: 'wrong_detection',
              interventionType,
              sessionId: feedbackSessionId,
            })
          }
          style={pillStyle}
        >
          Yanlis tespit
        </button>
        {interventionType === 'break' ? (
          <>
            <button
              onClick={() =>
                submitFeedback({
                  feedbackType: 'break_helpful',
                  interventionType,
                  sessionId: feedbackSessionId,
                })
              }
              style={pillStyle}
            >
              Mola ise yaradi
            </button>
            <button
              onClick={() =>
                submitFeedback({
                  feedbackType: 'break_not_helpful',
                  interventionType,
                  sessionId: feedbackSessionId,
                })
              }
              style={pillStyle}
            >
              Mola ise yaramadi
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() =>
                submitFeedback({
                  feedbackType: 'intervention_helpful',
                  interventionType,
                  sessionId: feedbackSessionId,
                })
              }
              style={pillStyle}
            >
              Mudahale ise yaradi
            </button>
            <button
              onClick={() =>
                submitFeedback({
                  feedbackType: 'intervention_not_helpful',
                  interventionType,
                  sessionId: feedbackSessionId,
                })
              }
              style={pillStyle}
            >
              Mudahale ise yaramadi
            </button>
          </>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: isMobile ? '18px' : '28px',
        background:
          'linear-gradient(135deg, #f3f7ff 0%, #f8fbff 42%, #eef8f1 100%)',
        fontFamily: 'Inter, sans-serif',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px',
      }}
    >
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '16px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <img
            src={logo}
            alt="FocusAI Logo"
            style={{ width: '46px', height: '46px', borderRadius: '14px' }}
          />
          <div>
            <h1 style={{ margin: 0, fontSize: '24px', color: '#0f172a' }}>
              FocusAI <span style={{ color: '#2563eb' }}>Assistant</span>
            </h1>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              Gecmis, analiz ve sureklilik fazi
            </div>
          </div>
        </div>
        <div
          style={{
            backgroundColor: '#fff7ed',
            color: '#9a3412',
            border: '1px solid #fdba74',
            padding: '8px 14px',
            borderRadius: '999px',
            fontWeight: 700,
            fontSize: '12px',
          }}
        >
          Kamera hook hazir, isleme kapali
        </div>
      </header>

      {(error || feedbackNotice) && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {error && (
            <div
              style={{
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                padding: '12px 14px',
                borderRadius: '14px',
                display: 'flex',
                justifyContent: 'space-between',
                gap: '10px',
              }}
            >
              <span>{error}</span>
              <button onClick={clearError} style={{ border: 'none', background: 'transparent' }}>
                Kapat
              </button>
            </div>
          )}
          {feedbackNotice && (
            <div
              style={{
                backgroundColor: '#ecfdf5',
                border: '1px solid #86efac',
                color: '#166534',
                padding: '12px 14px',
                borderRadius: '14px',
                display: 'flex',
                justifyContent: 'space-between',
                gap: '10px',
              }}
            >
              <span>{feedbackNotice}</span>
              <button
                onClick={clearFeedbackNotice}
                style={{ border: 'none', background: 'transparent' }}
              >
                Kapat
              </button>
            </div>
          )}
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 1.3fr) 390px',
          gap: '22px',
          alignItems: 'start',
        }}
      >
        <div
          style={{
            ...cardStyle,
            padding: 0,
            overflow: 'hidden',
            minHeight: isMobile ? 'auto' : 'calc(100vh - 130px)',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div
            style={{
              padding: '20px',
              borderBottom: '1px solid #e2e8f0',
              background:
                'linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.06))',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '12px',
              }}
            >
              <div>
                <h3 style={{ margin: 0, fontSize: '18px', color: '#0f172a' }}>AI Chatbox</h3>
                <div style={{ fontSize: '12px', color: '#64748b' }}>
                  Aktif oturum ve mentor mudahalesi
                </div>
              </div>
              <div
                style={{
                  backgroundColor: '#eff6ff',
                  color: currentBadgeColor,
                  padding: '9px 12px',
                  borderRadius: '999px',
                  fontWeight: 800,
                  fontSize: '12px',
                }}
              >
                {stateLabelMap[currentState]}
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr auto auto',
                gap: '10px',
              }}
            >
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Kullanici ID"
                disabled={!!sessionId || isSessionLoading}
                style={{ padding: '12px', borderRadius: '14px', border: '1px solid #cbd5e1' }}
              />
              <input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder={
                  welcomeData?.last_worked_topic
                    ? `Son konu: ${welcomeData.last_worked_topic}`
                    : 'Bugun hangi konuyu calisacaksin?'
                }
                disabled={!!sessionId || isSessionLoading}
                style={{ padding: '12px', borderRadius: '14px', border: '1px solid #cbd5e1' }}
              />
              <button
                onClick={handleSessionAction}
                disabled={isSessionLoading || (!sessionId && !userId.trim())}
                style={{
                  backgroundColor: sessionId ? '#dc2626' : '#1d4ed8',
                  color: 'white',
                  border: 'none',
                  borderRadius: '14px',
                  padding: '0 18px',
                  fontWeight: 700,
                  minHeight: '46px',
                }}
              >
                {isSessionLoading ? 'Bekleyin...' : sessionId ? 'Oturumu Kapat' : 'Oturum Baslat'}
              </button>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                style={{
                  backgroundColor: 'white',
                  color: '#0f172a',
                  border: '1px solid #cbd5e1',
                  borderRadius: '14px',
                  padding: '0 16px',
                  fontWeight: 700,
                  minHeight: '46px',
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
          </div>

          <div
            ref={messagesRef}
            style={{
              flex: 1,
              padding: '20px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            {messages.map((m, i) => (
              <div
                key={`${m.timestamp || i}-${i}`}
                style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '78%',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                }}
              >
                <div
                  style={{
                    backgroundColor: m.role === 'user' ? '#1d4ed8' : '#f8fafc',
                    color: m.role === 'user' ? 'white' : '#0f172a',
                    border: m.role === 'user' ? 'none' : '1px solid #e2e8f0',
                    padding: '13px 16px',
                    borderRadius: '18px',
                    lineHeight: '1.55',
                    whiteSpace: 'pre-wrap',
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
                      padding: '12px',
                      borderRadius: '14px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '10px',
                      fontSize: '13px',
                    }}
                  >
                    <div>
                      <strong>Mudahale ({m.mentorIntervention.intervention_type}): </strong>
                      {m.mentorIntervention.message}
                    </div>
                    {m.mentorIntervention.decision_reason && (
                      <div
                        style={{
                          backgroundColor: 'rgba(255,255,255,0.7)',
                          borderRadius: '12px',
                          padding: '10px',
                          lineHeight: '1.55',
                        }}
                      >
                        <strong>Secim nedeni: </strong>
                        {m.mentorIntervention.decision_reason}
                      </div>
                    )}
                    {renderFeedbackButtons(m.mentorIntervention.intervention_type)}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div
            style={{
              padding: '18px 20px 20px',
              borderTop: '1px solid #e2e8f0',
              display: 'flex',
              gap: '10px',
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
                padding: '14px 16px',
                borderRadius: '14px',
                border: '1px solid #cbd5e1',
              }}
            />
            <button
              onClick={handleSend}
              disabled={!sessionId || isLoading || isSessionLoading || !input.trim()}
              style={{
                backgroundColor: '#1d4ed8',
                color: 'white',
                border: 'none',
                borderRadius: '14px',
                padding: '0 24px',
                fontWeight: 700,
              }}
            >
              {isLoading ? 'Gonderiliyor...' : 'Gonder'}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div
            style={{
              ...cardStyle,
              background: 'linear-gradient(140deg, #1d4ed8 0%, #0f766e 100%)',
              color: 'white',
            }}
          >
            <div style={{ fontSize: '12px', opacity: 0.88, marginBottom: '10px' }}>
              BUGUNUN NOTU
            </div>
            <div style={{ fontSize: '13.5px', lineHeight: '1.6' }}>
              "{dailyQuotes[quoteIndex]}"
            </div>
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 700 }}>
              SUREKLILIK BLOKU
            </div>
            <h4 style={{ margin: '8px 0', fontSize: '18px', color: '#0f172a' }}>
              {welcomeData?.last_worked_topic || 'Yeni oturum hazir'}
            </h4>
            <div style={{ fontSize: '13px', color: '#334155', lineHeight: '1.6' }}>
              {welcomeData?.continue_suggestion ||
                'Son rapor, son konu ve devam onerisi burada gosterilecek.'}
            </div>
            <div
              style={{
                marginTop: '12px',
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '16px',
                padding: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '7px',
                fontSize: '13px',
                color: '#475569',
              }}
            >
              <div>Son neden: <strong>{welcomeData?.continue_reason || 'Henuz veri yok'}</strong></div>
              <div>Son session: <strong>{formatDateTime(welcomeData?.last_session?.ended_at)}</strong></div>
              <div>Son fokus: <strong>{formatPercent(welcomeData?.last_report?.focus_score ?? welcomeData?.last_session?.average_focus_score)}</strong></div>
              <div>Kisisel esik: <strong>{welcomeData?.baseline?.personalized_threshold?.toFixed(2) || '0.75'}</strong></div>
            </div>
            {welcomeData?.last_report?.summary_text && (
              <div
                style={{
                  marginTop: '12px',
                  backgroundColor: '#eef4ff',
                  color: '#1e3a8a',
                  borderRadius: '14px',
                  padding: '12px',
                  fontSize: '13px',
                  lineHeight: '1.6',
                }}
              >
                {welcomeData.last_report.summary_text}
              </div>
            )}
            {welcomeData?.latest_state_analysis?.reason_summary && (
              <div
                style={{
                  marginTop: '12px',
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '14px',
                  padding: '12px',
                  fontSize: '13px',
                  color: '#334155',
                  lineHeight: '1.6',
                }}
              >
                <strong>Son state karari: </strong>
                {welcomeData.latest_state_analysis.reason_summary}
              </div>
            )}
            {welcomeData?.latest_intervention?.reason && (
              <div
                style={{
                  marginTop: '12px',
                  backgroundColor: '#fff7ed',
                  border: '1px solid #fdba74',
                  borderRadius: '14px',
                  padding: '12px',
                  fontSize: '13px',
                  color: '#9a3412',
                  lineHeight: '1.6',
                }}
              >
                <strong>Son mudahale secimi: </strong>
                {welcomeData.latest_intervention.reason}
              </div>
            )}
            {welcomeData?.personalization_insights &&
              welcomeData.personalization_insights.length > 0 && (
                <div
                  style={{
                    marginTop: '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                  }}
                >
                  {welcomeData.personalization_insights.map((insight, index) => (
                    <div
                      key={`${insight}-${index}`}
                      style={{
                        backgroundColor: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        borderRadius: '14px',
                        padding: '10px 12px',
                        fontSize: '12px',
                        color: '#475569',
                        lineHeight: '1.55',
                      }}
                    >
                      {insight}
                    </div>
                  ))}
                </div>
              )}
          </div>

          <div style={cardStyle}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '12px',
              }}
            >
              <div>
                <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 700 }}>
                  ODAK TRENDI
                </div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#0f172a' }}>
                  {formatPercent(focusTrend?.average_focus_score)}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '12px', color: '#64748b' }}>Son 7 gun</div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#0f766e' }}>
                  {focusTrend?.total_sessions ?? 0}
                </div>
              </div>
            </div>
            <div style={{ height: '120px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={focusTrendChartData}>
                  <YAxis domain={[0, 100]} hide />
                  <Line type="monotone" dataKey="value" stroke="#0f766e" strokeWidth={3} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 700, marginBottom: '10px' }}>
              AKTIF OZET
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div style={{ backgroundColor: '#eff6ff', borderRadius: '14px', padding: '10px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>Toplam Odak</div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#0f172a' }}>
                  {dashboard ? `${dashboard.message_count * 2} dk` : stats.totalTime}
                </div>
              </div>
              <div style={{ backgroundColor: '#ecfdf5', borderRadius: '14px', padding: '10px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>Ort. Basari</div>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#0f172a' }}>
                  {dashboard ? formatPercent(dashboard.focus_score) : stats.avgSuccess}
                </div>
              </div>
            </div>
            <div style={{ height: '110px', marginTop: '12px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scores}>
                  <YAxis domain={[0, 100]} hide />
                  <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={3} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {dashboard && (
              <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13px', color: '#334155' }}>
                <div>Retry sayisi: <strong>{dashboard.retry_count}</strong></div>
                <div>Mudahale sayisi: <strong>{dashboard.intervention_count}</strong></div>
                <div>Konular: <strong>{dashboard.topics_covered.length ? dashboard.topics_covered.join(', ') : 'Yok'}</strong></div>
                {dashboard.latest_state_analysis?.reason_summary && (
                  <div
                    style={{
                      backgroundColor: '#eff6ff',
                      borderRadius: '14px',
                      padding: '12px',
                      lineHeight: '1.6',
                    }}
                  >
                    <strong>State nedeni: </strong>
                    {dashboard.latest_state_analysis.reason_summary}
                  </div>
                )}
                {dashboard.latest_intervention?.reason && (
                  <div
                    style={{
                      backgroundColor: '#fff7ed',
                      borderRadius: '14px',
                      padding: '12px',
                      lineHeight: '1.6',
                    }}
                  >
                    <strong>Mudahale secimi: </strong>
                    {dashboard.latest_intervention.reason}
                  </div>
                )}
                {dashboard.recommendations && dashboard.recommendations.length > 0 && (
                  <div
                    style={{
                      backgroundColor: '#f8fafc',
                      borderRadius: '14px',
                      padding: '12px',
                      lineHeight: '1.6',
                    }}
                  >
                    <strong>Oneriler: </strong>
                    {dashboard.recommendations.join(' | ')}
                  </div>
                )}
                {dashboard.summary_text && (
                  <div style={{ backgroundColor: '#f8fafc', borderRadius: '14px', padding: '12px', lineHeight: '1.6' }}>
                    {dashboard.summary_text}
                  </div>
                )}
              </div>
            )}
            {!dashboard && sessionSummary && (
              <div style={{ marginTop: '10px', fontSize: '13px', color: '#334155' }}>
                Hafiza kaydi: <strong>{sessionSummary.memory_entries_written}</strong>
              </div>
            )}
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '15px', color: '#0f172a', fontWeight: 700, marginBottom: '10px' }}>
              Gecmis Oturumlar
            </div>
            {sessionHistory.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '220px', overflowY: 'auto' }}>
                {sessionHistory.map((item) => (
                  <button
                    key={item.session_id}
                    onClick={() => void selectHistorySession(item.session_id)}
                    style={{
                      textAlign: 'left',
                      border: item.session_id === selectedHistorySessionId ? '1px solid #2563eb' : '1px solid #e2e8f0',
                      backgroundColor: item.session_id === selectedHistorySessionId ? '#eff6ff' : '#f8fafc',
                      borderRadius: '14px',
                      padding: '12px',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                    }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: 800, color: '#0f172a' }}>
                      {item.topic || 'Genel konu'}
                    </div>
                    <div style={{ fontSize: '12px', color: '#475569' }}>
                      {formatDateTime(item.ended_at || item.started_at)}
                    </div>
                    <div style={{ fontSize: '12px', color: '#475569' }}>
                      Mesaj: <strong>{item.message_count}</strong> | Fokus: <strong>{formatPercent(item.average_focus_score)}</strong>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>
                Bu kullanici icin session gecmisi bulunmuyor.
              </div>
            )}
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '15px', color: '#0f172a', fontWeight: 700, marginBottom: '10px' }}>
              Secili Mesaj Gecmisi
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '10px' }}>
              {selectedSession ? `${selectedSession.topic || 'Genel'} oturumu` : 'Bir oturum secin'}
            </div>
            {selectedHistoryMessages.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '260px', overflowY: 'auto' }}>
                {selectedHistoryMessages.map((message) => (
                  <div
                    key={message.id}
                    style={{
                      backgroundColor: message.role === 'user' ? '#eff6ff' : '#f8fafc',
                      border: '1px solid #e2e8f0',
                      borderRadius: '14px',
                      padding: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginBottom: '6px', fontSize: '11px', color: '#64748b' }}>
                      <strong style={{ color: '#334155' }}>{message.role}</strong>
                      <span>{formatDateTime(message.timestamp)}</span>
                    </div>
                    <div style={{ fontSize: '13px', color: '#0f172a', lineHeight: '1.55', whiteSpace: 'pre-wrap' }}>
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>
                Mesaj gecmisi burada gosterilecek.
              </div>
            )}
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: '15px', color: '#0f172a', fontWeight: 700, marginBottom: '10px' }}>
              Yuklenen PDF'ler
            </div>
            {uploadedDocuments.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {uploadedDocuments.map((doc) => (
                  <div
                    key={`${doc.filename}-${doc.uploaded_at}`}
                    style={{
                      backgroundColor: '#f8fafc',
                      border: '1px solid #e2e8f0',
                      borderRadius: '14px',
                      padding: '12px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '5px',
                    }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a' }}>
                      {doc.filename}
                    </div>
                    <div style={{ fontSize: '12px', color: '#475569' }}>
                      Parca: <strong>{doc.chunk_count}</strong> | Durum: <strong>{doc.indexed ? 'Hazir' : 'Hazir degil'}</strong>
                    </div>
                    <div style={{ fontSize: '12px', color: '#64748b' }}>
                      {formatDateTime(doc.uploaded_at)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>
                Bu kullanici icin yuklenmis PDF yok.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
