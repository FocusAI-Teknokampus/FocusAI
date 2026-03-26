import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts';
import logo from './logo.jpeg';
import { useFocusStore, type CameraPermissionState } from './store';

const stateLabelMap: Record<string, string> = {
  focused: 'Odakta',
  distracted: 'Daginik',
  fatigued: 'Yorgun',
  stuck: 'Takildi',
  unknown: 'Belirsiz',
};

const stateColorMap: Record<string, string> = {
  focused: '#15803d',
  distracted: '#c2410c',
  fatigued: '#b45309',
  stuck: '#b91c1c',
  unknown: '#475569',
};

const mentorModeLabelMap: Record<string, string> = {
  challenge: 'Challenge',
  guided_hint: 'Hint',
  direct_help: 'Direct',
  recovery: 'Recovery',
  clarify: 'Clarify',
};

const explanationStyleLabelMap: Record<string, string> = {
  brief: 'Kisa anlatim',
  detailed: 'Detayli anlatim',
  example_heavy: 'Ornekli anlatim',
};

const cardStyle: React.CSSProperties = {
  backgroundColor: 'rgba(255, 255, 255, 0.94)',
  border: '1px solid rgba(148, 163, 184, 0.18)',
  borderRadius: '28px',
  boxShadow: '0 18px 46px rgba(15, 23, 42, 0.08)',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '13px 14px',
  borderRadius: '16px',
  border: '1px solid #cbd5e1',
  backgroundColor: 'white',
  color: '#0f172a',
};

const sectionLabelStyle: React.CSSProperties = {
  fontSize: '11px',
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: '#64748b',
  fontWeight: 800,
};

function formatPercent(value?: number | null) {
  if (value == null) return 'Yok';
  return `%${Math.round(value * 100)}`;
}

function formatDateTime(value?: string | null) {
  if (!value) return 'Yok';
  return new Date(value).toLocaleString();
}

function firstText(...values: Array<string | null | undefined>) {
  return values.find((value) => Boolean(value && value.trim())) || '';
}

function getSignalLevel(value?: number | null) {
  if (value == null) return 'yok';
  if (value >= 0.7) return 'yuksek';
  if (value >= 0.4) return 'orta';
  if (value >= 0.18) return 'dusuk';
  return 'cok dusuk';
}

function getProbabilityEntries(probabilities?: Record<string, number> | null) {
  if (!probabilities) return [];
  return Object.entries(probabilities)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 3);
}

function getOutcomeLabel(item: {
  average_focus_score?: number | null;
  intervention_count: number;
  retry_count: number;
  summary_text?: string | null;
}) {
  const summary = (item.summary_text || '').toLowerCase();
  if (summary.includes('oturdu') || summary.includes('yerlesti')) return 'kavram oturdu';
  if ((item.average_focus_score ?? 0) >= 0.8 && item.intervention_count <= 2) {
    return 'kavram oturdu';
  }
  if ((item.average_focus_score ?? 0) >= 0.62) return 'ilerleme var';
  if (item.intervention_count >= 3 || item.retry_count >= 3) return 'tekrar gerekiyor';
  return 'yeniden bak';
}

function getProfileSummary(userProfile: {
  preferred_explanation_style: string;
  prefers_hint_first: boolean;
  prefers_direct_explanation: boolean;
  challenge_tolerance: number;
}) {
  const style =
    explanationStyleLabelMap[userProfile.preferred_explanation_style] ||
    'Standart anlatim';

  if (userProfile.prefers_direct_explanation) return `${style}, dogrudan yardim bekliyor`;
  if (userProfile.prefers_hint_first) return `${style}, once ipucuyla ilerliyor`;
  if (userProfile.challenge_tolerance >= 0.7) return `${style}, challenge tolere ediyor`;
  return style;
}

function getCameraPermissionLabel(permission: CameraPermissionState) {
  if (permission === 'granted') return 'Izin verildi';
  if (permission === 'denied') return 'Izin reddedildi';
  if (permission === 'error') return 'Kamera hatasi';
  return 'Izin bekleniyor';
}

function getCameraBackendLabel(cameraStatus?: {
  status?: string;
  active?: boolean;
  available?: boolean;
  face_detected?: boolean;
  backend_state?: string | null;
  attention_score?: number | null;
} | null) {
  if (!cameraStatus) return 'Backend beklemede';
  if (cameraStatus.status === 'error') return 'Backend hata verdi';
  if (!cameraStatus.active) return 'Backend bagli degil';
  if (!cameraStatus.available) return 'Frame geliyor, yuz araniyor';

  const focus = cameraStatus.attention_score != null
    ? ` | fokus ${Math.round(cameraStatus.attention_score * 100)}%`
    : '';
  const backendState = cameraStatus.backend_state
    ? ` | ${cameraStatus.backend_state.toLowerCase()}`
    : '';
  return cameraStatus.face_detected
    ? `CV aktif${focus}${backendState}`
    : 'CV aktif | yuz bekleniyor';
}

export default function App() {
  const {
    userId,
    learnerProfiles,
    userProfile,
    topic,
    sessionId,
    cameraPermission,
    cameraStreamActive,
    cameraStatus,
    currentState,
    messages,
    stats,
    dashboard,
    uploadedDocuments,
    sessionHistory,
    selectedHistorySessionId,
    selectedHistoryMessages,
    welcomeData,
    feedbackNotice,
    isLoading,
    isSessionLoading,
    error,
    setTopic,
    setCameraPermission,
    setCameraStreamActive,
    clearCameraStatus,
    selectLearnerProfile,
    createLearnerProfile,
    resumeLastProfile,
    clearError,
    clearFeedbackNotice,
    startSession,
    sendMessage,
    endSession,
    uploadPdf,
    hydrateUserWorkspace,
    pushCameraFrame,
    selectHistorySession,
    submitFeedback,
  } = useFocusStore();

  const [input, setInput] = useState('');
  const [newProfileName, setNewProfileName] = useState('');
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 1180);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameLoopRef = useRef<number | null>(null);
  const isUploadingFrameRef = useRef(false);

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
    if (!userId.trim()) return;
    void hydrateUserWorkspace(userId);
  }, [userId, hydrateUserWorkspace]);

  const stopCameraStream = () => {
    if (frameLoopRef.current != null) {
      window.clearInterval(frameLoopRef.current);
      frameLoopRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    isUploadingFrameRef.current = false;
    setCameraStreamActive(false);
    clearCameraStatus();
  };

  useEffect(() => () => stopCameraStream(), []);

  useEffect(() => {
    if (
      !sessionId ||
      cameraPermission !== 'granted' ||
      !cameraStreamActive ||
      !videoRef.current ||
      !canvasRef.current
    ) {
      if (frameLoopRef.current != null) {
        window.clearInterval(frameLoopRef.current);
        frameLoopRef.current = null;
      }
      return;
    }

    const captureAndSend = async () => {
      if (isUploadingFrameRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      isUploadingFrameRef.current = true;
      try {
        const targetWidth = 320;
        const targetHeight = Math.max(
          180,
          Math.round(targetWidth / Math.max(1, video.videoWidth || 16) * (video.videoHeight || 9))
        );

        canvas.width = targetWidth;
        canvas.height = targetHeight;

        const context = canvas.getContext('2d');
        if (!context) return;

        context.drawImage(video, 0, 0, targetWidth, targetHeight);
        await pushCameraFrame(canvas.toDataURL('image/jpeg', 0.72));
      } finally {
        isUploadingFrameRef.current = false;
      }
    };

    void captureAndSend();
    frameLoopRef.current = window.setInterval(() => {
      void captureAndSend();
    }, 1200);

    return () => {
      if (frameLoopRef.current != null) {
        window.clearInterval(frameLoopRef.current);
        frameLoopRef.current = null;
      }
    };
  }, [sessionId, cameraPermission, cameraStreamActive, pushCameraFrame]);

  const currentBadgeColor = useMemo(
    () => stateColorMap[currentState] || '#475569',
    [currentState]
  );
  const cameraPermissionLabel = getCameraPermissionLabel(cameraPermission);
  const cameraBackendLabel = getCameraBackendLabel(cameraStatus);

  const selectedSession = sessionHistory.find(
    (item) => item.session_id === selectedHistorySessionId
  );
  const feedbackSessionId =
    sessionId || welcomeData?.last_session?.session_id || selectedHistorySessionId;
  const currentMentorMessage = [...messages]
    .reverse()
    .find(
      (message) =>
        message.role === 'ai' &&
        Boolean(
          message.mentorMode ||
            message.mentorIntervention ||
            message.ragSource ||
            message.currentState
        )
    );
  const lastFocusScore =
    dashboard?.focus_score ??
    welcomeData?.last_report?.focus_score ??
    welcomeData?.last_session?.average_focus_score;
  const openLoop = firstText(
    welcomeData?.last_struggling_concept,
    welcomeData?.continue_reason,
    welcomeData?.last_report?.weaknesses?.[0]
  );
  const todayStart = firstText(
    welcomeData?.today_start_recommendation,
    welcomeData?.operational_next_session_plan?.first_prompt,
    welcomeData?.mini_recall_question,
    'Bugun once hedefini tek cumleyle yaz.'
  );
  const continueSuggestion = firstText(
    welcomeData?.continue_suggestion,
    welcomeData?.operational_next_session_plan?.start_with,
    'Kaldigin yerden devam etmek icin ilk alt adimi sec.'
  );
  const lastMentorReason = firstText(
    currentMentorMessage?.mentorReasons?.[0],
    currentMentorMessage?.mentorIntervention?.decision_reason,
    dashboard?.latest_state_analysis?.reason_summary,
    welcomeData?.latest_state_analysis?.reason_summary
  );
  const lastMentorSource = firstText(
    currentMentorMessage?.ragSource,
    uploadedDocuments[0]?.filename
  );
  const lastMentorMode = firstText(
    currentMentorMessage?.mentorMode,
    welcomeData?.operational_next_session_plan?.mentor_tactic,
    userProfile?.best_intervention_type
  );
  const latestStateAnalysis =
    dashboard?.latest_state_analysis ?? welcomeData?.latest_state_analysis ?? null;
  const latestFeatureVector = latestStateAnalysis?.feature_vector ?? null;
  const fatigueTextScore = latestFeatureVector?.fatigue_text_score ?? null;
  const confusionScore = latestFeatureVector?.confusion_score ?? null;
  const semanticRetryScore = latestFeatureVector?.semantic_retry_score ?? null;
  const dominantSignals = latestStateAnalysis?.dominant_signals ?? currentMentorMessage?.dominantSignals ?? [];
  const topStateProbabilities = getProbabilityEntries(latestStateAnalysis?.state_probabilities ?? null);
  const recentBalanceData = sessionHistory
    .slice(0, 7)
    .reverse()
    .map((item, index) => ({
      label: item.topic ? item.topic.slice(0, 12) : `S${index + 1}`,
      value: Math.round((item.average_focus_score ?? 0) * 100),
    }));

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading || isSessionLoading) return;
    setInput('');
    await sendMessage(trimmed);
  };

  const handleCameraAccess = async () => {
    if (cameraStreamActive) {
      stopCameraStream();
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraPermission('error');
      stopCameraStream();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }

      setCameraPermission('granted');
      setCameraStreamActive(true);
    } catch (err) {
      const errorName = err instanceof DOMException ? err.name : '';
      setCameraPermission(
        errorName === 'NotAllowedError' || errorName === 'SecurityError'
          ? 'denied'
          : 'error'
      );
      stopCameraStream();
    }
  };

  const handleSessionAction = async () => {
    if (sessionId) {
      await endSession();
      stopCameraStream();
      return;
    }
    await startSession(cameraPermission === 'granted' && cameraStreamActive);
  };

  const handlePdfSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await uploadPdf(file);
    event.target.value = '';
  };

  const handleCreateProfile = async () => {
    const trimmed = newProfileName.trim();
    if (!trimmed) return;
    setNewProfileName('');
    await createLearnerProfile(trimmed);
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
        <button onClick={() => submitFeedback({ feedbackType: 'correct_detection', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Dogru tespit</button>
        <button onClick={() => submitFeedback({ feedbackType: 'wrong_detection', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Yanlis tespit</button>
        {interventionType === 'break' ? (
          <>
            <button onClick={() => submitFeedback({ feedbackType: 'break_helpful', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Mola ise yaradi</button>
            <button onClick={() => submitFeedback({ feedbackType: 'break_not_helpful', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Mola ise yaramadi</button>
          </>
        ) : (
          <>
            <button onClick={() => submitFeedback({ feedbackType: 'intervention_helpful', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Mudahale ise yaradi</button>
            <button onClick={() => submitFeedback({ feedbackType: 'intervention_not_helpful', interventionType, sessionId: feedbackSessionId })} style={pillStyle}>Mudahale ise yaramadi</button>
          </>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: isMobile ? '18px' : '30px',
        background:
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), linear-gradient(140deg, #eef6ff 0%, #f7fbff 48%, #f3faf4 100%)',
        fontFamily: '"Segoe UI Variable Text", "Segoe UI", sans-serif',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px',
      }}
    >
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '18px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <img src={logo} alt="FocusAI Logo" style={{ width: '50px', height: '50px', borderRadius: '16px' }} />
          <div>
            <h1 style={{ margin: 0, fontSize: '26px', color: '#0f172a' }}>
              FocusAI <span style={{ color: '#0f766e' }}>Mentor Workspace</span>
            </h1>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              Bilgi mimarisi odakli calisma akisi
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <div style={{ backgroundColor: '#eff6ff', color: currentBadgeColor, border: `1px solid ${currentBadgeColor}22`, padding: '9px 14px', borderRadius: '999px', fontWeight: 800, fontSize: '12px' }}>
            {stateLabelMap[currentState]}
          </div>
          <div style={{ backgroundColor: cameraPermission === 'granted' ? '#ecfdf5' : '#fff7ed', color: cameraPermission === 'granted' ? '#166534' : '#9a3412', border: `1px solid ${cameraPermission === 'granted' ? '#86efac' : '#fdba74'}`, padding: '9px 14px', borderRadius: '999px', fontWeight: 700, fontSize: '12px' }}>
            Kamera izni: {cameraPermissionLabel}
          </div>
          <div style={{ backgroundColor: cameraStatus?.active ? '#eff6ff' : '#f8fafc', color: cameraStatus?.active ? '#1d4ed8' : '#475569', border: `1px solid ${cameraStatus?.active ? '#93c5fd' : '#cbd5e1'}`, padding: '9px 14px', borderRadius: '999px', fontWeight: 700, fontSize: '12px' }}>
            {cameraBackendLabel}
          </div>
        </div>
      </header>

      {(error || feedbackNotice) && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {error && (
            <div style={{ backgroundColor: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', padding: '12px 14px', borderRadius: '16px', display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
              <span>{error}</span>
              <button onClick={clearError} style={{ border: 'none', background: 'transparent', color: 'inherit' }}>Kapat</button>
            </div>
          )}
          {feedbackNotice && (
            <div style={{ backgroundColor: '#ecfdf5', border: '1px solid #86efac', color: '#166534', padding: '12px 14px', borderRadius: '16px', display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
              <span>{feedbackNotice}</span>
              <button onClick={clearFeedbackNotice} style={{ border: 'none', background: 'transparent', color: 'inherit' }}>Kapat</button>
            </div>
          )}
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 1.45fr) 360px',
          gap: '22px',
          alignItems: 'start',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div
            style={{
              ...cardStyle,
              padding: 0,
              overflow: 'hidden',
              minHeight: isMobile ? 'auto' : 'calc(100vh - 148px)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                padding: '22px',
                borderBottom: '1px solid #e2e8f0',
                background:
                  'linear-gradient(135deg, rgba(15,118,110,0.08), rgba(14,165,233,0.08))',
                display: 'flex',
                flexDirection: 'column',
                gap: '14px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', flexWrap: 'wrap' }}>
                <div>
                  <div style={sectionLabelStyle}>Calisma Oturumu</div>
                  <h3 style={{ margin: '6px 0 4px', fontSize: '20px', color: '#0f172a' }}>Mentor ile canli ilerleme</h3>
                  <div style={{ fontSize: '13px', color: '#475569' }}>
                    Profil sec, konuyu belirle, sonra mentorun hangi modda cevap verdigini gor.
                  </div>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'flex-end' }}>
                  <div style={{ backgroundColor: 'white', border: '1px solid #dbeafe', borderRadius: '999px', padding: '8px 12px', fontSize: '12px', color: '#1e3a8a', fontWeight: 700 }}>
                    Profil: {userId}
                  </div>
                  <div style={{ backgroundColor: 'white', border: '1px solid #d1fae5', borderRadius: '999px', padding: '8px 12px', fontSize: '12px', color: '#065f46', fontWeight: 700 }}>
                    Fokus: {dashboard ? formatPercent(dashboard.focus_score) : stats.avgSuccess}
                  </div>
                  <div style={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '999px', padding: '8px 12px', fontSize: '12px', color: '#334155', fontWeight: 700 }}>
                    Mudahale: {dashboard?.intervention_count ?? 0}
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 1.1fr) auto minmax(0, 1fr) auto', gap: '10px' }}>
                <select value={userId} onChange={(event) => void selectLearnerProfile(event.target.value)} disabled={!!sessionId || isSessionLoading} style={inputStyle}>
                  {learnerProfiles.map((profile) => (
                    <option key={profile.id} value={profile.id}>
                      {profile.label} ({profile.id})
                    </option>
                  ))}
                </select>
                <button onClick={() => void resumeLastProfile()} disabled={!!sessionId || isSessionLoading} style={{ border: '1px solid #cbd5e1', borderRadius: '16px', backgroundColor: 'white', padding: '0 16px', fontWeight: 700, minHeight: '48px' }}>
                  Son kullanilan
                </button>
                <input value={newProfileName} onChange={(event) => setNewProfileName(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && !sessionId && !isSessionLoading ? void handleCreateProfile() : undefined} placeholder="Yeni profil olustur" disabled={!!sessionId || isSessionLoading} style={inputStyle} />
                <button onClick={() => void handleCreateProfile()} disabled={!!sessionId || isSessionLoading || !newProfileName.trim()} style={{ border: 'none', borderRadius: '16px', backgroundColor: '#0f766e', color: 'white', padding: '0 18px', fontWeight: 700, minHeight: '48px' }}>
                  Yeni profil
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'auto minmax(0, 1fr)', gap: '10px', alignItems: 'stretch' }}>
                <button onClick={() => void handleCameraAccess()} style={{ border: 'none', borderRadius: '16px', backgroundColor: cameraStreamActive ? '#b91c1c' : '#0f766e', color: 'white', padding: '0 18px', fontWeight: 800, minHeight: '48px' }}>
                  {cameraStreamActive ? 'Kamerayi Kapat' : 'Kamera Izni Ver'}
                </button>
                <div style={{ backgroundColor: 'rgba(255,255,255,0.85)', border: '1px solid #dbeafe', borderRadius: '16px', padding: '12px 14px', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '4px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: '#1e3a8a' }}>
                    {cameraPermissionLabel} {cameraStreamActive ? '| stream acik' : '| stream kapali'}
                  </div>
                  <div style={{ fontSize: '12px', color: cameraStatus?.status === 'error' ? '#991b1b' : '#475569' }}>
                    {cameraStatus?.error || cameraBackendLabel}
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 1fr) auto auto', gap: '10px' }}>
                <input value={topic} onChange={(event) => setTopic(event.target.value)} placeholder={welcomeData?.last_worked_topic ? `Son konu: ${welcomeData.last_worked_topic}` : 'Bugun hangi konuyu calisacaksin?'} disabled={!!sessionId || isSessionLoading} style={inputStyle} />
                <button onClick={handleSessionAction} disabled={isSessionLoading || (!sessionId && !userId.trim())} style={{ border: 'none', borderRadius: '16px', backgroundColor: sessionId ? '#dc2626' : '#1d4ed8', color: 'white', padding: '0 22px', fontWeight: 800, minHeight: '48px' }}>
                  {isSessionLoading ? 'Bekleyin...' : sessionId ? 'Oturumu Kapat' : 'Oturum Baslat'}
                </button>
                <button onClick={() => fileInputRef.current?.click()} disabled={isLoading} style={{ border: '1px solid #cbd5e1', borderRadius: '16px', backgroundColor: 'white', color: '#0f172a', padding: '0 16px', fontWeight: 700, minHeight: '48px' }}>
                  PDF Yukle
                </button>
                <input ref={fileInputRef} type="file" accept="application/pdf" onChange={handlePdfSelected} style={{ display: 'none' }} />
              </div>
            </div>

            <div ref={messagesRef} style={{ flex: 1, padding: '22px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {messages.map((message, index) => {
                const reason = message.mentorReasons?.[0] || message.mentorIntervention?.decision_reason || '';
                const modeLabel = message.mentorMode ? mentorModeLabelMap[message.mentorMode] || message.mentorMode : '';
                const sourceLabel = message.ragSource || '';

                return (
                  <div key={`${message.timestamp || index}-${index}`} style={{ alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {message.role === 'ai' && (modeLabel || reason || sourceLabel) && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', padding: '10px 12px', borderRadius: '16px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', fontSize: '12px', color: '#334155' }}>
                        {modeLabel && <span><strong>Mentor modu:</strong> {modeLabel}</span>}
                        {reason && <span><strong>Neden:</strong> {reason}</span>}
                        {sourceLabel && <span><strong>Kaynak:</strong> {sourceLabel}</span>}
                      </div>
                    )}
                    <div style={{ backgroundColor: message.role === 'user' ? '#0f766e' : '#ffffff', color: message.role === 'user' ? 'white' : '#0f172a', border: message.role === 'user' ? 'none' : '1px solid rgba(148, 163, 184, 0.2)', padding: '14px 16px', borderRadius: '20px', lineHeight: '1.6', whiteSpace: 'pre-wrap', boxShadow: message.role === 'user' ? '0 12px 28px rgba(15, 118, 110, 0.18)' : '0 10px 24px rgba(15, 23, 42, 0.05)' }}>
                      <strong>{message.role === 'ai' ? 'mentor: ' : 'sen: '}</strong>
                      {message.text}
                    </div>
                    {message.role === 'ai' && message.currentState && (
                      <div style={{ alignSelf: 'flex-start', backgroundColor: '#f8fafc', color: stateColorMap[message.currentState] || '#475569', border: `1px solid ${stateColorMap[message.currentState] || '#cbd5e1'}`, padding: '6px 10px', borderRadius: '999px', fontSize: '12px', fontWeight: 700 }}>
                        Durum: {stateLabelMap[message.currentState]}
                      </div>
                    )}
                    {message.role === 'ai' && message.mentorIntervention && (
                      <div style={{ alignSelf: 'flex-start', backgroundColor: '#fff7ed', color: '#9a3412', border: '1px solid #fdba74', padding: '12px', borderRadius: '16px', display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                        <div><strong>Mudahale ({message.mentorIntervention.intervention_type}):</strong> {message.mentorIntervention.message}</div>
                        {message.mentorIntervention.decision_reason && (
                          <div style={{ backgroundColor: 'rgba(255,255,255,0.72)', borderRadius: '14px', padding: '10px', lineHeight: '1.55' }}>
                            <strong>Secim nedeni:</strong> {message.mentorIntervention.decision_reason}
                          </div>
                        )}
                        {renderFeedbackButtons(message.mentorIntervention.intervention_type)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div style={{ padding: '18px 22px 22px', borderTop: '1px solid #e2e8f0', display: 'flex', gap: '10px' }}>
              <input value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && void handleSend()} placeholder={sessionId ? 'Mesajini yaz...' : 'Once oturum baslat'} disabled={!sessionId || isLoading || isSessionLoading} style={{ ...inputStyle, flex: 1 }} />
              <button onClick={() => void handleSend()} disabled={!sessionId || isLoading || isSessionLoading || !input.trim()} style={{ border: 'none', borderRadius: '16px', backgroundColor: '#1d4ed8', color: 'white', padding: '0 24px', fontWeight: 800 }}>
                {isLoading ? 'Gonderiliyor...' : 'Gonder'}
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 0.9fr) minmax(0, 1.1fr)', gap: '16px' }}>
            <div style={{ ...cardStyle, padding: '20px' }}>
              <div style={sectionLabelStyle}>Calisma Dengesi</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: '10px', margin: '8px 0 4px' }}>
                <h3 style={{ margin: 0, fontSize: '20px', color: '#0f172a' }}>Son 7 oturum calisma dengesi</h3>
                <div style={{ fontSize: '22px', fontWeight: 800, color: '#0f766e' }}>
                  {formatPercent(recentBalanceData.length ? recentBalanceData.reduce((sum, item) => sum + item.value, 0) / recentBalanceData.length / 100 : null)}
                </div>
              </div>
              <div style={{ fontSize: '13px', color: '#475569', lineHeight: '1.55' }}>
                Her oturumdaki ortalama odak skorunun kisa denge ozeti.
              </div>
              <div style={{ height: '150px', marginTop: '16px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={recentBalanceData}>
                    <YAxis domain={[0, 100]} hide />
                    <Line type="monotone" dataKey="value" stroke="#0f766e" strokeWidth={3} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div style={{ ...cardStyle, padding: '20px' }}>
              <div style={sectionLabelStyle}>Gecmis Oturumlar</div>
              <h3 style={{ margin: '8px 0 4px', fontSize: '20px', color: '#0f172a' }}>Gecmisi anlamli sinyallerle oku</h3>
              <div style={{ fontSize: '13px', color: '#475569', marginBottom: '12px' }}>
                Konu, odak, mudahale ve sonuc etiketi birlikte gorunur.
              </div>
              {sessionHistory.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '280px', overflowY: 'auto' }}>
                  {sessionHistory.map((item) => (
                    <button key={item.session_id} onClick={() => void selectHistorySession(item.session_id)} style={{ textAlign: 'left', border: item.session_id === selectedHistorySessionId ? '1px solid #0f766e' : '1px solid #e2e8f0', backgroundColor: item.session_id === selectedHistorySessionId ? '#f0fdfa' : '#f8fafc', borderRadius: '18px', padding: '14px', cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
                        <strong style={{ fontSize: '14px', color: '#0f172a' }}>{item.topic || 'Genel konu'}</strong>
                        <span style={{ backgroundColor: '#e0f2fe', color: '#075985', borderRadius: '999px', padding: '4px 10px', fontSize: '11px', fontWeight: 700 }}>
                          Sonuc: {getOutcomeLabel(item)}
                        </span>
                      </div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>{formatDateTime(item.ended_at || item.started_at)}</div>
                      <div style={{ fontSize: '12px', color: '#334155' }}>{formatPercent(item.average_focus_score)} odak</div>
                      <div style={{ fontSize: '12px', color: '#334155' }}>{item.intervention_count} mentor mudahalesi</div>
                    </button>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: '13px', color: '#64748b' }}>Bu profil icin oturum gecmisi yok.</div>
              )}
            </div>
          </div>

          <div style={{ ...cardStyle, padding: '20px' }}>
            <div style={sectionLabelStyle}>Secili Oturum</div>
            <h3 style={{ margin: '8px 0 4px', fontSize: '20px', color: '#0f172a' }}>Mesaj akisi</h3>
            <div style={{ fontSize: '13px', color: '#475569', marginBottom: '12px' }}>
              {selectedSession ? `${selectedSession.topic || 'Genel konu'} oturumu` : 'Bir oturum secildiginde detay burada gorunur.'}
            </div>
            {selectedHistoryMessages.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '320px', overflowY: 'auto' }}>
                {selectedHistoryMessages.map((message) => (
                  <div key={message.id} style={{ backgroundColor: message.role === 'user' ? '#eff6ff' : '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '18px', padding: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginBottom: '6px', fontSize: '11px', color: '#64748b' }}>
                      <strong style={{ color: '#334155' }}>{message.role}</strong>
                      <span>{formatDateTime(message.timestamp)}</span>
                    </div>
                    <div style={{ fontSize: '13px', color: '#0f172a', lineHeight: '1.55', whiteSpace: 'pre-wrap' }}>{message.content}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>Mesaj gecmisi burada gosterilecek.</div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ ...cardStyle, padding: '20px' }}>
            <div style={sectionLabelStyle}>Nerede Kalmistin?</div>
            <h3 style={{ margin: '8px 0 14px', fontSize: '20px', color: '#0f172a' }}>
              {welcomeData?.last_worked_topic || topic || 'Yeni baslangic'}
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ backgroundColor: '#eff6ff', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Devam onerisi</div>
                <div style={{ marginTop: '6px', fontSize: '14px', color: '#0f172a' }}>{continueSuggestion}</div>
              </div>
              <div style={{ backgroundColor: '#fff7ed', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Gecen oturumdan kalan acik</div>
                <div style={{ marginTop: '6px', fontSize: '14px', color: '#7c2d12' }}>
                  {openLoop || 'Acik kalan belirgin bir nokta yok.'}
                </div>
              </div>
              <div style={{ backgroundColor: '#ecfdf5', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Bugun icin baslangic</div>
                <div style={{ marginTop: '6px', fontSize: '14px', color: '#065f46' }}>{todayStart}</div>
              </div>
            </div>
            <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div style={{ backgroundColor: '#f8fafc', borderRadius: '16px', padding: '12px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>Son odak</div>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#0f172a' }}>{formatPercent(lastFocusScore)}</div>
              </div>
              <div style={{ backgroundColor: '#f8fafc', borderRadius: '16px', padding: '12px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>Son oturum</div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a' }}>{formatDateTime(welcomeData?.last_session?.ended_at)}</div>
              </div>
            </div>
          </div>

          <div style={{ ...cardStyle, padding: '20px' }}>
            <div style={sectionLabelStyle}>Mentor Icgorusu</div>
            <h3 style={{ margin: '8px 0 14px', fontSize: '20px', color: '#0f172a' }}>Sistem nasil karar veriyor?</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ backgroundColor: '#f8fafc', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Aktif mentor modu</div>
                <div style={{ marginTop: '6px', fontSize: '16px', fontWeight: 800, color: '#0f172a' }}>
                  {mentorModeLabelMap[lastMentorMode] || lastMentorMode || 'Hazirda bekliyor'}
                </div>
              </div>
              <div style={{ backgroundColor: '#eff6ff', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Neden</div>
                <div style={{ marginTop: '6px', fontSize: '14px', color: '#1e3a8a' }}>
                  {lastMentorReason || 'Belirgin bir state gerekcesi kaydi yok.'}
                </div>
              </div>
              <div style={{ backgroundColor: '#fff7ed', borderRadius: '18px', padding: '14px' }}>
                <div style={sectionLabelStyle}>Kaynak</div>
                <div style={{ marginTop: '6px', fontSize: '14px', color: '#9a3412' }}>
                  {lastMentorSource || 'Bu cevapta not kaynagi kullanilmadi.'}
                </div>
              </div>
              {(fatigueTextScore != null || confusionScore != null || semanticRetryScore != null || topStateProbabilities.length > 0 || dominantSignals.length > 0) && (
                <div style={{ backgroundColor: '#f8fafc', borderRadius: '18px', padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={sectionLabelStyle}>Son state snapshot</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                    <div style={{ fontSize: '13px', color: '#0f172a' }}>
                      Yorgunluk ifadesi: <strong>{getSignalLevel(fatigueTextScore)}</strong>
                    </div>
                    <div style={{ fontSize: '13px', color: '#0f172a' }}>
                      Karisiklik: <strong>{getSignalLevel(confusionScore)}</strong>
                    </div>
                    <div style={{ fontSize: '13px', color: '#0f172a' }}>
                      Tekrar: <strong>{getSignalLevel(semanticRetryScore)}</strong>
                    </div>
                  </div>
                  {dominantSignals.length > 0 && (
                    <div style={{ fontSize: '12px', color: '#475569', lineHeight: '1.55' }}>
                      Dominant signals: {dominantSignals.join(', ')}
                    </div>
                  )}
                  {topStateProbabilities.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {topStateProbabilities.map(([stateKey, value]) => (
                        <div key={stateKey} style={{ backgroundColor: 'white', border: '1px solid #dbeafe', borderRadius: '999px', padding: '6px 10px', fontSize: '12px', color: '#1e3a8a', fontWeight: 700 }}>
                          {stateLabelMap[stateKey] || stateKey}: {formatPercent(value)}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {dashboard?.latest_intervention?.reason && (
                <div style={{ backgroundColor: '#f8fafc', borderRadius: '18px', padding: '14px' }}>
                  <div style={sectionLabelStyle}>Son mudahale secimi</div>
                  <div style={{ marginTop: '6px', fontSize: '14px', color: '#334155' }}>{dashboard.latest_intervention.reason}</div>
                </div>
              )}
            </div>
          </div>

          <div style={{ ...cardStyle, padding: '20px' }}>
            <div style={sectionLabelStyle}>Calisma Profili</div>
            <h3 style={{ margin: '8px 0 14px', fontSize: '20px', color: '#0f172a' }}>Ogrencinin ritmi</h3>
            <div style={{ backgroundColor: '#f8fafc', borderRadius: '18px', padding: '14px', marginBottom: '10px' }}>
              <div style={{ fontSize: '14px', color: '#0f172a', fontWeight: 700 }}>
                {userProfile ? getProfileSummary(userProfile) : 'Profil verisi yukleniyor'}
              </div>
              <div style={{ marginTop: '6px', fontSize: '12px', color: '#64748b' }}>
                Kisiye ozel esik: <strong>{userProfile?.adaptive_threshold?.toFixed(2) || welcomeData?.baseline.personalized_threshold?.toFixed(2) || '0.75'}</strong>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
              <div style={{ backgroundColor: '#eff6ff', borderRadius: '16px', padding: '12px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>Toplam oturum</div>
                <div style={{ fontSize: '20px', fontWeight: 800, color: '#0f172a' }}>{userProfile?.total_sessions ?? 0}</div>
              </div>
              <div style={{ backgroundColor: '#ecfdf5', borderRadius: '16px', padding: '12px' }}>
                <div style={{ fontSize: '11px', color: '#64748b' }}>En iyi mudahale</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: '#0f172a' }}>
                  {userProfile?.best_intervention_type || welcomeData?.intervention_policy.best_intervention_type || 'Yok'}
                </div>
              </div>
            </div>
            {(userProfile?.frequent_struggle_topics?.length || userProfile?.strong_topics?.length || welcomeData?.personalization_insights?.length) && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {userProfile?.frequent_struggle_topics?.slice(0, 2).map((topicItem) => (
                  <div key={topicItem} style={{ backgroundColor: '#fff7ed', borderRadius: '14px', padding: '10px 12px', fontSize: '12px', color: '#9a3412' }}>
                    Zorlanilan konu: {topicItem}
                  </div>
                ))}
                {userProfile?.strong_topics?.slice(0, 2).map((topicItem) => (
                  <div key={topicItem} style={{ backgroundColor: '#ecfdf5', borderRadius: '14px', padding: '10px 12px', fontSize: '12px', color: '#166534' }}>
                    Guclu alan: {topicItem}
                  </div>
                ))}
                {welcomeData?.personalization_insights?.slice(0, 2).map((insight) => (
                  <div key={insight} style={{ backgroundColor: '#f8fafc', borderRadius: '14px', padding: '10px 12px', fontSize: '12px', color: '#475569', lineHeight: '1.55' }}>
                    {insight}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ ...cardStyle, padding: '20px' }}>
            <div style={sectionLabelStyle}>Yuklenen Notlar</div>
            <h3 style={{ margin: '8px 0 14px', fontSize: '20px', color: '#0f172a' }}>Mentorun basvurdugu kaynaklar</h3>
            {uploadedDocuments.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {uploadedDocuments.map((document) => (
                  <div key={`${document.filename}-${document.uploaded_at}`} style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '18px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a' }}>{document.filename}</div>
                    <div style={{ fontSize: '12px', color: '#475569' }}>{document.chunk_count} parca | <strong>{document.indexed ? 'Hazir' : 'Hazir degil'}</strong></div>
                    <div style={{ fontSize: '12px', color: '#64748b' }}>{formatDateTime(document.uploaded_at)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: '#64748b' }}>Bu profil icin yuklenmis not yok.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
