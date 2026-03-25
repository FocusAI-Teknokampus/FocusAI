import { create } from "zustand";

const API_BASE = "http://127.0.0.1:8000";

type UserState = "focused" | "distracted" | "fatigued" | "stuck" | "unknown";
type InterventionType = "hint" | "strategy" | "break" | "mode_switch" | "question" | "none";

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
}

interface ProfileResponse {
  user_id: string;
  preferred_explanation_style: string;
  weak_topics: string[];
  strong_topics: string[];
  recurring_misconceptions: string[];
  adaptive_threshold: number;
  total_sessions: number;
}

interface UploadResponse {
  user_id: string;
  filename: string;
  chunk_count: number;
  indexed: boolean;
  message: string;
}

interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  currentState?: UserState;
  mentorIntervention?: MentorIntervention | null;
}

interface FocusStore {
  userId: string;
  topic: string;
  sessionId: string | null;
  messages: UIMessage[];
  currentState: UserState;
  dashboard: DashboardResponse | null;
  profile: ProfileResponse | null;
  sessionSummary: SessionEndResponse | null;
  isLoading: boolean;
  error: string | null;

  setUserId: (userId: string) => void;
  setTopic: (topic: string) => void;
  clearError: () => void;

  startSession: (userId: string, topic?: string, cameraEnabled?: boolean) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  endSession: () => Promise<void>;
  uploadPdf: (file: File) => Promise<UploadResponse | null>;
  fetchDashboard: (sessionId: string) => Promise<void>;
  fetchProfile: (userId: string) => Promise<void>;
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.detail || "Bir hata oluştu.";
  } catch {
    return "Bir hata oluştu.";
  }
}

export const useFocusStore = create<FocusStore>((set, get) => ({
  userId: "user_001",
  topic: "",
  sessionId: null,
  messages: [],
  currentState: "unknown",
  dashboard: null,
  profile: null,
  sessionSummary: null,
  isLoading: false,
  error: null,

  setUserId: (userId) => set({ userId }),
  setTopic: (topic) => set({ topic }),
  clearError: () => set({ error: null }),

  startSession: async (userId, topic = "", cameraEnabled = false) => {
    set({ isLoading: true, error: null });

    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          topic: topic || null,
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
        topic: data.topic ?? "",
        messages: [],
        currentState: "unknown",
        sessionSummary: null,
      });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Oturum başlatılamadı." });
    } finally {
      set({ isLoading: false });
    }
  },

  sendMessage: async (content) => {
    const { sessionId, userId, messages } = get();

    if (!sessionId) {
      set({ error: "Önce oturum başlatmalısın." });
      return;
    }

    const userMessage: UIMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    };

    set({
      isLoading: true,
      error: null,
      messages: [...messages, userMessage],
    });

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
          content,
          channel: "text",
        }),
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: ChatApiResponse = await res.json();

      const assistantMessage: UIMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.content,
        timestamp: data.timestamp,
        currentState: data.current_state,
        mentorIntervention: data.mentor_intervention ?? null,
      };

      set((state) => ({
        messages: [...state.messages, assistantMessage],
        currentState: data.current_state,
      }));
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Mesaj gönderilemedi." });
    } finally {
      set({ isLoading: false });
    }
  },

  endSession: async () => {
    const { sessionId, userId } = get();

    if (!sessionId) {
      set({ error: "Kapatılacak aktif oturum yok." });
      return;
    }

    set({ isLoading: true, error: null });

    try {
      const res = await fetch(`${API_BASE}/session/end`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
        }),
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: SessionEndResponse = await res.json();

      set({
        sessionSummary: data,
        sessionId: null,
      });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Oturum kapatılamadı." });
    } finally {
      set({ isLoading: false });
    }
  },

  uploadPdf: async (file) => {
    const { userId } = get();
    set({ isLoading: true, error: null });

    try {
      const formData = new FormData();
      formData.append("user_id", userId);
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: UploadResponse = await res.json();
      return data;
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "PDF yüklenemedi." });
      return null;
    } finally {
      set({ isLoading: false });
    }
  },

  fetchDashboard: async (sessionId) => {
    set({ isLoading: true, error: null });

    try {
      const res = await fetch(`${API_BASE}/dashboard/${sessionId}`);
      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: DashboardResponse = await res.json();
      set({ dashboard: data });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Dashboard verisi alınamadı." });
    } finally {
      set({ isLoading: false });
    }
  },

  fetchProfile: async (userId) => {
    set({ isLoading: true, error: null });

    try {
      const res = await fetch(`${API_BASE}/dashboard/profile/${userId}`);
      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: ProfileResponse = await res.json();
      set({ profile: data });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : "Profil verisi alınamadı." });
    } finally {
      set({ isLoading: false });
    }
  },
}));