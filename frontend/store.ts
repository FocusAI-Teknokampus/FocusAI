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
  };
}

interface UploadResponse {
  user_id: string;
  filename: string;
  chunk_count: number;
  indexed: boolean;
  message: string;
}

export interface ChatMessageUI {
  role: 'ai' | 'user';
  text: string;
  currentState?: UserState;
  mentorIntervention?: MentorIntervention | null;
  ragSource?: string | null;
  timestamp?: string;
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

  isLoading: boolean;
  isSessionLoading: boolean;
  error: string | null;

  setUserId: (userId: string) => void;
  setTopic: (topic: string) => void;
  clearError: () => void;

  addScore: (score: number) => void;
  startSession: (cameraEnabled?: boolean) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  endSession: () => Promise<void>;
  uploadPdf: (file: File) => Promise<void>;
  fetchDashboard: (targetSessionId?: string) => Promise<void>;
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.detail || 'Bir hata oluştu.';
  } catch {
    return 'Bir hata oluştu.';
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
  // Backend daha zengin ve nested bir dashboard modeli donuyor.
  // Frontend bu ekranda yalnizca temel ozet alanlarini kullandigi icin
  // burada tek bir ceviri noktasi olusturuyoruz.
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
      text: 'Selam! Bugün hangi konuyu çalışıyoruz? Sana ders notlarından yardımcı olabilirim.',
    },
  ],

  scores: [],
  stats: { totalTime: '0 dk', avgSuccess: '%0' },

  sessionSummary: null,
  dashboard: null,

  isLoading: false,
  isSessionLoading: false,
  error: null,

  setUserId: (userId) => set({ userId }),
  setTopic: (topic) => set({ topic }),
  clearError: () => set({ error: null }),

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
    const { userId, topic } = get();

    if (!userId.trim()) {
      set({ error: 'Kullanıcı ID gerekli.' });
      return;
    }

    set({
      isSessionLoading: true,
      error: null,
      sessionSummary: null,
      dashboard: null,
    });

    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId.trim(),
          topic: topic.trim() || null,
          // Kullanici tercihini backend'e gonderiyoruz.
          // Boylesiyle session kaydi kamera kullanildi bilgisini dogru tutar.
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
            text: 'Oturum başlatıldı. Hazırsan ilk sorunu yazabilirsin.',
          },
        ],
        scores: [],
        stats: { totalTime: '0 dk', avgSuccess: '%0' },
      });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Oturum başlatılamadı.',
      });
    } finally {
      set({ isSessionLoading: false });
    }
  },

  sendMessage: async (content) => {
    const { sessionId, userId, messages, scores } = get();

    if (!sessionId) {
      set({ error: 'Önce oturum başlat.' });
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
          error: 'Oturum artık geçerli değil. Lütfen yeniden oturum başlat.',
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

      // Dashboard endpoint'i aktif oturumda RAM fallback kullandigi icin
      // her mesajdan sonra sag paneldeki ozet verileri guncel kalir.
      await get().fetchDashboard(sessionId);
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Mesaj gönderilemedi.',
      });
    } finally {
      set({ isLoading: false });
    }
  },

  endSession: async () => {
    const { sessionId, userId } = get();

    if (!sessionId) {
      set({ error: 'Kapatılacak aktif oturum yok.' });
      return;
    }

    set({
      isSessionLoading: true,
      error: null,
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
          error: 'Oturum backend tarafında bulunamadı. Yerel oturum temizlendi.',
        });
        return;
      }

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: SessionEndResponse = await res.json();

      // Session kapanir kapanmaz rapor DB tarafinda olusuyor.
      // sessionId temizlenmeden ayni oturumun dashboard verisini cekiyoruz.
      await get().fetchDashboard(sessionId);

      set({
        sessionSummary: data,
        sessionId: null,
        currentState: 'unknown',
      });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Oturum kapatılamadı.',
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
            text: `PDF yüklendi: ${data.filename} (${data.chunk_count} parça indekslendi)`,
          },
        ],
      }));
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'PDF yüklenemedi.',
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
      // sessiz geç
    }
  },
}));
