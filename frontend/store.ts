import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8000';

type UserState = 'focused' | 'distracted' | 'fatigued' | 'stuck' | 'unknown';

type InterventionType =
  | 'hint'
  | 'strategy'
  | 'break'
  | 'mode_switch'
  | 'question'
  | 'none';

interface MentorIntervention {
  intervention_type: InterventionType;
  message: string;
  triggered_by: UserState;
  confidence: number;
  timestamp: string;
  decision_reason?: string | null;
  policy_snapshot?: Record<string, unknown>;
}

interface ChatApiResponse {
  session_id: string;
  content: string;
  rag_source?: string | null;
  mentor_intervention?: MentorIntervention | null;
  current_state: UserState;
  timestamp: string;
}

interface SessionStartResponse {
  session_id: string;
  user_id: string;
  topic?: string | null;
  camera_enabled: boolean;
  started_at: string;
}

interface SessionEndResponse {
  status: string;
  memory_entries_written: number;
  topics_covered: string[];
}

interface DashboardResponse {
  session_id: string;
  user_id: string;
  topic?: string | null;
  message_count: number;
  topics_covered: string[];
  current_state: UserState;
  retry_count: number;
  intervention_count: number;
  focus_score: number | null;
  summary_text: string | null;
  latest_state_analysis?: {
    state_after?: string | null;
    confidence?: number | null;
    threshold?: number | null;
    decision_margin?: number | null;
    uncertainty_signal?: number | null;
    learning_pattern?: string | null;
    reason_summary?: string | null;
    deviation_features?: Record<string, unknown>;
    state_scores?: Record<string, number>;
    state_probabilities?: Record<string, number>;
  } | null;
  latest_intervention?: {
    intervention_type?: string | null;
    message?: string | null;
    reason?: string | null;
    confidence?: number | null;
    was_successful?: boolean | null;
    timestamp?: string | null;
  } | null;
  strengths?: string[];
  weaknesses?: string[];
  recommendations?: string[];
  next_session_plan?: Record<string, unknown> | null;
  source?: string | null;
}

interface DashboardApiResponse {
  session_id: string;
  user_id: string;
  topic?: string | null;
  current_state?: string | null;
  intervention_count?: number;
  average_focus_score?: number | null;
  source?: string | null;
  retry_count?: number;
  report?: {
    message_count?: number;
    topics_covered?: string[];
    retry_count?: number;
    intervention_count?: number;
    focus_score?: number | null;
    summary_text?: string | null;
    strengths?: string[];
    weaknesses?: string[];
    recommendations?: string[];
    next_session_plan?: Record<string, unknown>;
  };
  latest_state_analysis?: DashboardResponse['latest_state_analysis'];
  latest_intervention?: DashboardResponse['latest_intervention'];
}

interface UploadResponse {
  user_id: string;
  filename: string;
  chunk_count: number;
  indexed: boolean;
  message: string;
}

interface UploadedDocumentSummary {
  filename: string;
  file_type?: string | null;
  file_size_bytes?: number | null;
  chunk_count: number;
  indexed: boolean;
  uploaded_at: string;
}

interface SessionHistoryItem {
  session_id: string;
  user_id: string;
  topic?: string | null;
  subtopic?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  current_state?: string | null;
  average_focus_score?: number | null;
  retry_count: number;
  intervention_count: number;
  message_count: number;
  summary_text?: string | null;
}

interface SessionHistoryMessage {
  id: string;
  session_id: string;
  role: string;
  content: string;
  timestamp?: string | null;
  user_state?: string | null;
  detected_topic?: string | null;
  message_type?: string | null;
  llm_confidence?: number | null;
}

interface FocusTrendPoint {
  date: string;
  focus_score: number;
  session_count: number;
}

interface FocusTrendResponse {
  user_id: string;
  days: number;
  total_sessions: number;
  average_focus_score: number | null;
  trend_direction: 'up' | 'down' | 'stable';
  points: FocusTrendPoint[];
}

interface BaselineSummary {
  user_id: string;
  sample_session_count: number;
  enough_data: boolean;
  avg_message_length: number;
  avg_response_time_seconds: number;
  avg_idle_gap_seconds: number;
  avg_messages_per_session: number;
  avg_session_duration_seconds: number;
  avg_focus_score: number | null;
  question_style?: string | null;
  personalized_threshold: number;
  metrics?: Record<string, unknown>;
  updated_at?: string | null;
}

interface InterventionPolicySummaryItem {
  user_id: string;
  intervention_type: string;
  total_count: number;
  success_count: number;
  failure_count: number;
  success_rate?: number | null;
  recent_success_rate?: number | null;
  states?: string[];
  last_feedback_type?: string | null;
  last_outcome?: boolean | null;
  updated_at?: string | null;
}

interface WelcomeResponse {
  user_id: string;
  has_history: boolean;
  last_session?: {
    session_id: string;
    topic?: string | null;
    subtopic?: string | null;
    started_at?: string | null;
    ended_at?: string | null;
    current_state?: string | null;
    average_focus_score?: number | null;
    intervention_count: number;
    retry_count: number;
  } | null;
  last_report?: {
    summary_text?: string | null;
    focus_score?: number | null;
    message_count: number;
    intervention_count: number;
    retry_count: number;
    topics_covered: string[];
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
    next_session_plan?: {
      goal?: string;
      suggested_duration_minutes?: number;
      recommended_actions?: string[];
    } | null;
    created_at?: string | null;
  } | null;
  last_worked_topic?: string | null;
  continue_suggestion: string;
  continue_reason: string;
  baseline: BaselineSummary;
  latest_state_analysis?: DashboardResponse['latest_state_analysis'];
  latest_intervention?: DashboardResponse['latest_intervention'];
  personalization_insights?: string[];
  intervention_policy: {
    best_intervention_type?: string | null;
    items: InterventionPolicySummaryItem[];
  };
}

interface FeedbackApiResponse {
  status: string;
  feedback_id: string;
  adaptive_threshold?: number | null;
  intervention_type?: string | null;
  intervention_success_rate?: number | null;
}

export interface ChatMessageUI {
  role: 'ai' | 'user';
  text: string;
  currentState?: UserState;
  mentorIntervention?: MentorIntervention | null;
  ragSource?: string | null;
  timestamp?: string;
}

interface SubmitFeedbackParams {
  feedbackType: string;
  interventionType?: string | null;
  sessionId?: string | null;
  messageId?: string | null;
  notes?: string;
}

interface FocusState {
  userId: string;
  topic: string;
  sessionId: string | null;
  currentState: UserState;
  messages: ChatMessageUI[];
  scores: { time: string; value: number }[];
  stats: { totalTime: string; avgSuccess: string };
  sessionSummary: SessionEndResponse | null;
  dashboard: DashboardResponse | null;
  uploadedDocuments: UploadedDocumentSummary[];
  sessionHistory: SessionHistoryItem[];
  selectedHistorySessionId: string | null;
  selectedHistoryMessages: SessionHistoryMessage[];
  focusTrend: FocusTrendResponse | null;
  welcomeData: WelcomeResponse | null;
  feedbackNotice: string | null;
  isLoading: boolean;
  isSessionLoading: boolean;
  error: string | null;
  setUserId: (userId: string) => void;
  setTopic: (topic: string) => void;
  clearError: () => void;
  clearFeedbackNotice: () => void;
  addScore: (score: number) => void;
  startSession: (cameraEnabled?: boolean) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  endSession: () => Promise<void>;
  uploadPdf: (file: File) => Promise<void>;
  fetchDashboard: (targetSessionId?: string) => Promise<void>;
  fetchUploadedDocuments: (targetUserId?: string) => Promise<void>;
  fetchSessionHistory: (targetUserId?: string) => Promise<void>;
  fetchSessionMessages: (targetSessionId: string) => Promise<void>;
  fetchFocusTrend: (targetUserId?: string, days?: number) => Promise<void>;
  fetchWelcome: (targetUserId?: string) => Promise<void>;
  hydrateUserWorkspace: (targetUserId?: string) => Promise<void>;
  selectHistorySession: (sessionId: string) => Promise<void>;
  submitFeedback: (params: SubmitFeedbackParams) => Promise<void>;
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.detail || 'Bir hata olustu.';
  } catch {
    return 'Bir hata olustu.';
  }
}

function toUiState(state?: string | null): UserState {
  const normalized = (state || 'unknown').toLowerCase();
  if (
    normalized === 'focused' ||
    normalized === 'distracted' ||
    normalized === 'fatigued' ||
    normalized === 'stuck' ||
    normalized === 'unknown'
  ) {
    return normalized;
  }
  return 'unknown';
}

function calcAvgSuccess(scores: { value: number; time: string }[]) {
  if (scores.length === 0) return '%0';
  const avg = Math.round(
    scores.reduce((sum, item) => sum + item.value, 0) / scores.length
  );
  return `%${avg}`;
}

function calcTotalTime(messageCount: number) {
  return `${messageCount * 2} dk`;
}

function mapDashboardResponse(data: DashboardApiResponse): DashboardResponse {
  return {
    session_id: data.session_id,
    user_id: data.user_id,
    topic: data.topic ?? null,
    message_count: data.report?.message_count ?? 0,
    topics_covered: data.report?.topics_covered ?? [],
    current_state: toUiState(data.current_state),
    retry_count: data.report?.retry_count ?? data.retry_count ?? 0,
    intervention_count:
      data.report?.intervention_count ?? data.intervention_count ?? 0,
    focus_score: data.report?.focus_score ?? data.average_focus_score ?? null,
    summary_text: data.report?.summary_text ?? null,
    latest_state_analysis: data.latest_state_analysis ?? null,
    latest_intervention: data.latest_intervention ?? null,
    strengths: data.report?.strengths ?? [],
    weaknesses: data.report?.weaknesses ?? [],
    recommendations: data.report?.recommendations ?? [],
    next_session_plan: data.report?.next_session_plan ?? null,
    source: data.source ?? null,
  };
}

export const useFocusStore = create<FocusState>((set, get) => ({
  userId: 'user_001',
  topic: '',
  sessionId: null,
  currentState: 'unknown',
  messages: [
    {
      role: 'ai',
      text: 'Selam! Bugun hangi konuyu calisiyoruz? Sana ders notlarindan yardimci olabilirim.',
    },
  ],
  scores: [],
  stats: { totalTime: '0 dk', avgSuccess: '%0' },
  sessionSummary: null,
  dashboard: null,
  uploadedDocuments: [],
  sessionHistory: [],
  selectedHistorySessionId: null,
  selectedHistoryMessages: [],
  focusTrend: null,
  welcomeData: null,
  feedbackNotice: null,
  isLoading: false,
  isSessionLoading: false,
  error: null,

  setUserId: (userId) => set({ userId }),
  setTopic: (topic) => set({ topic }),
  clearError: () => set({ error: null }),
  clearFeedbackNotice: () => set({ feedbackNotice: null }),

  addScore: (score) =>
    set((state) => {
      const nextScores = [
        ...state.scores,
        {
          time: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          }),
          value: score,
        },
      ].slice(-15);

      return {
        scores: nextScores,
        stats: {
          ...state.stats,
          avgSuccess: calcAvgSuccess(nextScores),
        },
      };
    }),

  startSession: async (cameraEnabled = false) => {
    const { userId, topic, welcomeData } = get();
    const trimmedUserId = userId.trim();

    if (!trimmedUserId) {
      set({ error: 'Kullanici ID gerekli.' });
      return;
    }

    const fallbackTopic = welcomeData?.last_worked_topic?.trim() || '';

    set({
      isSessionLoading: true,
      error: null,
      sessionSummary: null,
      dashboard: null,
      feedbackNotice: null,
    });

    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: trimmedUserId,
          topic: topic.trim() || fallbackTopic || null,
          camera_enabled: cameraEnabled,
        }),
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: SessionStartResponse = await res.json();

      set({
        sessionId: data.session_id,
        userId: data.user_id,
        topic: data.topic ?? '',
        currentState: 'unknown',
        messages: [
          {
            role: 'ai',
            text: 'Oturum baslatildi. Hazirsan ilk sorunu yazabilirsin.',
          },
        ],
        scores: [],
        stats: { totalTime: '0 dk', avgSuccess: '%0' },
      });

      await Promise.all([
        get().fetchUploadedDocuments(data.user_id),
        get().fetchSessionHistory(data.user_id),
        get().fetchWelcome(data.user_id),
      ]);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Oturum baslatilamadi.',
      });
    } finally {
      set({ isSessionLoading: false });
    }
  },

  sendMessage: async (content) => {
    const { sessionId, userId, messages, scores } = get();

    if (!sessionId) {
      set({ error: 'Once oturum baslat.' });
      return;
    }

    const trimmed = content.trim();
    if (!trimmed) return;

    const userMessage: ChatMessageUI = {
      role: 'user',
      text: trimmed,
      timestamp: new Date().toISOString(),
    };

    set({
      isLoading: true,
      error: null,
      feedbackNotice: null,
      messages: [...messages, userMessage],
    });

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
          content: trimmed,
          channel: 'text',
        }),
      });

      if (res.status === 404) {
        set({
          sessionId: null,
          currentState: 'unknown',
          error: 'Oturum artik gecerli degil. Lutfen yeniden oturum baslat.',
        });
        return;
      }

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: ChatApiResponse = await res.json();
      const nextState = toUiState(data.current_state);

      const stateScoreMap: Record<UserState, number> = {
        focused: 88,
        distracted: 42,
        fatigued: 33,
        stuck: 25,
        unknown: 55,
      };

      const nextScores = [
        ...scores,
        {
          time: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          }),
          value: stateScoreMap[nextState],
        },
      ].slice(-15);

      const aiMessage: ChatMessageUI = {
        role: 'ai',
        text: data.content,
        currentState: nextState,
        mentorIntervention: data.mentor_intervention ?? null,
        ragSource: data.rag_source ?? null,
        timestamp: data.timestamp,
      };

      set((state) => ({
        messages: [...state.messages, aiMessage],
        currentState: nextState,
        scores: nextScores,
        stats: {
          totalTime: calcTotalTime(
            [...state.messages, aiMessage].filter((m) => m.role === 'user').length
          ),
          avgSuccess: calcAvgSuccess(nextScores),
        },
      }));

      await get().fetchDashboard(sessionId);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Mesaj gonderilemedi.',
      });
    } finally {
      set({ isLoading: false });
    }
  },

  endSession: async () => {
    const { sessionId, userId } = get();

    if (!sessionId) {
      set({ error: 'Kapatilacak aktif oturum yok.' });
      return;
    }

    set({
      isSessionLoading: true,
      error: null,
      feedbackNotice: null,
    });

    try {
      const res = await fetch(`${API_BASE}/session/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
        }),
      });

      if (res.status === 404) {
        set({
          sessionId: null,
          currentState: 'unknown',
          error: 'Oturum backend tarafinda bulunamadi. Yerel oturum temizlendi.',
        });
        return;
      }

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: SessionEndResponse = await res.json();
      await get().fetchDashboard(sessionId);

      set({
        sessionSummary: data,
        sessionId: null,
        currentState: 'unknown',
      });

      await get().hydrateUserWorkspace(userId);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Oturum kapatilamadi.',
      });
    } finally {
      set({ isSessionLoading: false });
    }
  },

  uploadPdf: async (file) => {
    const { userId } = get();

    set({ isLoading: true, error: null });

    try {
      const formData = new FormData();
      formData.append('user_id', userId);
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: UploadResponse = await res.json();

      set((state) => ({
        messages: [
          ...state.messages,
          {
            role: 'ai',
            text: `PDF yuklendi: ${data.filename} (${data.chunk_count} parca indekslendi)`,
          },
        ],
      }));
      await get().fetchUploadedDocuments(userId);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'PDF yuklenemedi.',
      });
    } finally {
      set({ isLoading: false });
    }
  },

  fetchDashboard: async (targetSessionId) => {
    const { sessionId } = get();
    const activeSessionId = targetSessionId ?? sessionId;

    if (!activeSessionId) return;

    try {
      const res = await fetch(`${API_BASE}/dashboard/${activeSessionId}`);
      if (!res.ok) return;

      const data: DashboardApiResponse = await res.json();
      set({ dashboard: mapDashboardResponse(data) });
    } catch {
      // sessiz gec
    }
  },

  fetchUploadedDocuments: async (targetUserId) => {
    const { userId } = get();
    const activeUserId = (targetUserId ?? userId).trim();

    if (!activeUserId) {
      set({ uploadedDocuments: [] });
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/upload/documents/${encodeURIComponent(activeUserId)}`
      );
      if (!res.ok) return;

      const data: UploadedDocumentSummary[] = await res.json();
      set({ uploadedDocuments: data });
    } catch {
      // yardimci liste
    }
  },

  fetchSessionHistory: async (targetUserId) => {
    const { userId, selectedHistorySessionId } = get();
    const activeUserId = (targetUserId ?? userId).trim();

    if (!activeUserId) {
      set({
        sessionHistory: [],
        selectedHistorySessionId: null,
        selectedHistoryMessages: [],
      });
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/history/sessions/${encodeURIComponent(activeUserId)}`
      );
      if (!res.ok) return;

      const data: SessionHistoryItem[] = await res.json();
      const nextSelected =
        selectedHistorySessionId &&
        data.some((item) => item.session_id === selectedHistorySessionId)
          ? selectedHistorySessionId
          : data[0]?.session_id ?? null;

      set({
        sessionHistory: data,
        selectedHistorySessionId: nextSelected,
      });

      if (nextSelected) {
        await get().fetchSessionMessages(nextSelected);
      } else {
        set({ selectedHistoryMessages: [] });
      }
    } catch {
      // yardimci veri
    }
  },

  fetchSessionMessages: async (targetSessionId) => {
    if (!targetSessionId) {
      set({ selectedHistoryMessages: [] });
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/history/session/${encodeURIComponent(targetSessionId)}/messages`
      );
      if (!res.ok) return;

      const data: SessionHistoryMessage[] = await res.json();
      set({
        selectedHistorySessionId: targetSessionId,
        selectedHistoryMessages: data,
      });
    } catch {
      // yardimci veri
    }
  },

  fetchFocusTrend: async (targetUserId, days = 7) => {
    const { userId } = get();
    const activeUserId = (targetUserId ?? userId).trim();

    if (!activeUserId) {
      set({ focusTrend: null });
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/analytics/focus-trend/${encodeURIComponent(activeUserId)}?days=${days}`
      );
      if (!res.ok) return;

      const data: FocusTrendResponse = await res.json();
      set({ focusTrend: data });
    } catch {
      // yardimci veri
    }
  },

  fetchWelcome: async (targetUserId) => {
    const { userId } = get();
    const activeUserId = (targetUserId ?? userId).trim();

    if (!activeUserId) {
      set({ welcomeData: null });
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/welcome/${encodeURIComponent(activeUserId)}`);
      if (!res.ok) return;

      const data: WelcomeResponse = await res.json();
      set({ welcomeData: data });
    } catch {
      // yardimci veri
    }
  },

  hydrateUserWorkspace: async (targetUserId) => {
    const { userId } = get();
    const activeUserId = (targetUserId ?? userId).trim();

    if (!activeUserId) {
      set({
        uploadedDocuments: [],
        sessionHistory: [],
        selectedHistorySessionId: null,
        selectedHistoryMessages: [],
        focusTrend: null,
        welcomeData: null,
      });
      return;
    }

    await Promise.all([
      get().fetchUploadedDocuments(activeUserId),
      get().fetchSessionHistory(activeUserId),
      get().fetchFocusTrend(activeUserId),
      get().fetchWelcome(activeUserId),
    ]);
  },

  selectHistorySession: async (targetSessionId) => {
    await get().fetchSessionMessages(targetSessionId);
  },

  submitFeedback: async ({
    feedbackType,
    interventionType,
    sessionId,
    messageId,
    notes,
  }) => {
    const { userId } = get();
    const activeUserId = userId.trim();

    if (!activeUserId) {
      set({ error: 'Feedback gondermek icin kullanici ID gerekli.' });
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: activeUserId,
          session_id: sessionId ?? null,
          message_id: messageId ?? null,
          feedback_type: feedbackType,
          target_type: 'intervention',
          intervention_type: interventionType ?? null,
          notes: notes ?? null,
        }),
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: FeedbackApiResponse = await res.json();
      const thresholdText =
        data.adaptive_threshold != null
          ? ` Yeni esik: ${data.adaptive_threshold.toFixed(2)}`
          : '';

      set({
        feedbackNotice: `Feedback kaydedildi.${thresholdText}`,
        error: null,
      });

      await get().fetchWelcome(activeUserId);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Feedback gonderilemedi.',
      });
    }
  },
}));
