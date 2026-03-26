import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8000';

type UserState = 'focused' | 'distracted' | 'fatigued' | 'stuck' | 'unknown';

type ResponsePolicyMode =
  | 'direct_help'
  | 'guided_hint'
  | 'challenge'
  | 'recovery'
  | 'clarify';

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
  response_policy?: ResponsePolicyMode | null;
  response_reasons?: string[];
  dominant_signals?: string[];
  policy_path?: string[];
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

export type CameraPermissionState = 'idle' | 'granted' | 'denied' | 'error';

interface CameraSignalPayload {
  timestamp: string;
  ear_score: number;
  gaze_on_screen: boolean;
  hand_on_chin: boolean;
  head_tilt_angle?: number | null;
}

interface CameraStatusResponse {
  session_id: string;
  status: 'idle' | 'active' | 'error' | string;
  available: boolean;
  active: boolean;
  face_detected: boolean;
  backend_state?: string | null;
  attention_score?: number | null;
  processing_ms?: number | null;
  frame_id: number;
  signal?: CameraSignalPayload | null;
  last_updated_at?: string | null;
  error?: string | null;
}

interface DashboardResponse {
  session_id: string;
  user_id: string;
  topic?: string | null;
  subtopic?: string | null;
  study_mode?: string | null;
  camera_used: boolean;
  started_at?: string | null;
  ended_at?: string | null;
  message_count: number;
  topics_covered: string[];
  current_state: UserState;
  retry_count: number;
  intervention_count: number;
  focus_score: number | null;
  focus_timeline: {
    timestamp: string;
    score: number;
    state: UserState | string;
  }[];
  behavior_timeline: Record<string, unknown>[];
  behavior_summary?: Record<string, unknown>;
  summary_text: string | null;
  latest_state_analysis?: {
    state_after?: string | null;
    confidence?: number | null;
    threshold?: number | null;
    decision_margin?: number | null;
    uncertainty_signal?: number | null;
    learning_pattern?: string | null;
    response_policy?: ResponsePolicyMode | null;
    dominant_signals?: string[];
    reasons?: string[];
    reason_summary?: string | null;
    deviation_features?: Record<string, unknown>;
    state_scores?: Record<string, number>;
    state_probabilities?: Record<string, number>;
    feature_vector?: {
      fatigue_text_score?: number | null;
      confusion_score?: number | null;
      semantic_retry_score?: number | null;
      help_seeking_score?: number | null;
      answer_commitment_score?: number | null;
      [key: string]: unknown;
    } | null;
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
  subtopic?: string | null;
  study_mode?: string | null;
  camera_used?: boolean;
  started_at?: string | null;
  ended_at?: string | null;
  current_state?: string | null;
  intervention_count?: number;
  average_focus_score?: number | null;
  focus_timeline?: DashboardResponse['focus_timeline'];
  behavior_timeline?: Record<string, unknown>[];
  source?: string | null;
  retry_count?: number;
  report?: {
    message_count?: number;
    topics_covered?: string[];
    retry_count?: number;
    intervention_count?: number;
    focus_score?: number | null;
    summary_text?: string | null;
    behavior_summary?: Record<string, unknown>;
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
  resume_topic?: string | null;
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
  last_struggling_concept?: string | null;
  continue_suggestion: string;
  today_start_recommendation?: string;
  continue_reason: string;
  mini_recall_question?: string | null;
  baseline: BaselineSummary;
  latest_state_analysis?: DashboardResponse['latest_state_analysis'];
  latest_intervention?: DashboardResponse['latest_intervention'];
  latest_feedback_impact?: {
    timestamp?: string | null;
    intervention_type?: string | null;
    measurement_status?: string | null;
    [key: string]: unknown;
  } | null;
  operational_next_session_plan?: {
    topic?: string | null;
    subtopic?: string | null;
    duration_minutes?: number | null;
    start_with?: string | null;
    first_prompt?: string | null;
    target_outcome?: string | null;
    why_now?: string | null;
    mentor_tactic?: string | null;
    session_structure?: string[];
    checkpoints?: string[];
    risk_watchouts?: string[];
    state_carryover?: string | null;
    feedback_carryover?: string | null;
  } | null;
  personalization_insights?: string[];
  intervention_policy: {
    best_intervention_type?: string | null;
    items: InterventionPolicySummaryItem[];
  };
}

interface UserProfileResponse {
  user_id: string;
  preferred_explanation_style: string;
  weak_topics: string[];
  strong_topics: string[];
  recurring_misconceptions: string[];
  adaptive_threshold: number;
  total_sessions: number;
  normal_message_length: number;
  normal_response_delay_seconds: number;
  typical_retry_level: number;
  frequent_struggle_topics: string[];
  best_intervention_type?: string | null;
  prefers_hint_first: boolean;
  prefers_direct_explanation: boolean;
  challenge_tolerance: number;
  intervention_sensitivity: number;
}

interface LearnerProfileOption {
  id: string;
  label: string;
  createdAt: string;
  lastUsedAt?: string | null;
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
  mentorMode?: ResponsePolicyMode | null;
  mentorReasons?: string[];
  dominantSignals?: string[];
  policyPath?: string[];
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
  learnerProfiles: LearnerProfileOption[];
  userProfile: UserProfileResponse | null;
  topic: string;
  sessionId: string | null;
  cameraPermission: CameraPermissionState;
  cameraStreamActive: boolean;
  cameraStatus: CameraStatusResponse | null;
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
  setCameraPermission: (permission: CameraPermissionState) => void;
  setCameraStreamActive: (active: boolean) => void;
  clearCameraStatus: () => void;
  selectLearnerProfile: (profileId: string) => Promise<void>;
  createLearnerProfile: (label: string) => Promise<void>;
  resumeLastProfile: () => Promise<void>;
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
  fetchWelcome: (targetUserId?: string, topicOverride?: string) => Promise<void>;
  fetchUserProfile: (targetUserId?: string) => Promise<void>;
  hydrateUserWorkspace: (targetUserId?: string) => Promise<void>;
  selectHistorySession: (sessionId: string) => Promise<void>;
  pushCameraFrame: (imageBase64: string) => Promise<void>;
  submitFeedback: (params: SubmitFeedbackParams) => Promise<void>;
}

const PROFILE_STORAGE_KEY = 'focusai.learner_profiles';
const LAST_PROFILE_STORAGE_KEY = 'focusai.last_profile_id';
const DEFAULT_USER_ID = 'user_001';

function defaultAiMessage(): ChatMessageUI {
  return {
    role: 'ai',
    text: 'Selam! Bugun hangi konuyu calisiyoruz? Sana ders notlarindan yardimci olabilirim.',
  };
}

function normalizeProfileId(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/ı/g, 'i')
    .replace(/ğ/g, 'g')
    .replace(/ü/g, 'u')
    .replace(/ş/g, 's')
    .replace(/ö/g, 'o')
    .replace(/ç/g, 'c')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function sortProfiles(profiles: LearnerProfileOption[]) {
  return [...profiles].sort((a, b) => {
    const lastA = a.lastUsedAt ? new Date(a.lastUsedAt).getTime() : 0;
    const lastB = b.lastUsedAt ? new Date(b.lastUsedAt).getTime() : 0;
    if (lastA !== lastB) return lastB - lastA;
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
  });
}

function readStoredProfiles(): LearnerProfileOption[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .filter(
        (item): item is LearnerProfileOption =>
          typeof item?.id === 'string' &&
          typeof item?.label === 'string' &&
          typeof item?.createdAt === 'string'
      )
      .map((item) => ({
        id: normalizeProfileId(item.id),
        label: item.label.trim() || item.id,
        createdAt: item.createdAt,
        lastUsedAt: item.lastUsedAt ?? null,
      }))
      .filter((item) => item.id);
  } catch {
    return [];
  }
}

function persistProfiles(profiles: LearnerProfileOption[]) {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(
    PROFILE_STORAGE_KEY,
    JSON.stringify(sortProfiles(profiles))
  );
}

function readLastProfileId() {
  if (typeof window === 'undefined') {
    return null;
  }
  const value = window.localStorage.getItem(LAST_PROFILE_STORAGE_KEY);
  return value ? normalizeProfileId(value) : null;
}

function upsertProfile(
  profiles: LearnerProfileOption[],
  profileId: string,
  label?: string,
  markAsUsed = false
) {
  const normalizedId = normalizeProfileId(profileId);
  if (!normalizedId) return sortProfiles(profiles);

  const now = new Date().toISOString();
  const existing = profiles.find((item) => item.id === normalizedId);
  const next = existing
    ? profiles.map((item) =>
        item.id === normalizedId
          ? {
              ...item,
              label: label?.trim() || item.label,
              lastUsedAt: markAsUsed ? now : item.lastUsedAt ?? null,
            }
          : item
      )
    : [
        ...profiles,
        {
          id: normalizedId,
          label: label?.trim() || normalizedId,
          createdAt: now,
          lastUsedAt: markAsUsed ? now : null,
        },
      ];

  const sorted = sortProfiles(next);
  persistProfiles(sorted);

  if (markAsUsed && typeof window !== 'undefined') {
    window.localStorage.setItem(LAST_PROFILE_STORAGE_KEY, normalizedId);
  }

  return sorted;
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
    subtopic: data.subtopic ?? null,
    study_mode: data.study_mode ?? null,
    camera_used: data.camera_used ?? false,
    started_at: data.started_at ?? null,
    ended_at: data.ended_at ?? null,
    message_count: data.report?.message_count ?? 0,
    topics_covered: data.report?.topics_covered ?? [],
    current_state: toUiState(data.current_state),
    retry_count: data.report?.retry_count ?? data.retry_count ?? 0,
    intervention_count:
      data.report?.intervention_count ?? data.intervention_count ?? 0,
    focus_score: data.report?.focus_score ?? data.average_focus_score ?? null,
    focus_timeline: data.focus_timeline ?? [],
    behavior_timeline: data.behavior_timeline ?? [],
    behavior_summary: data.report?.behavior_summary ?? {},
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

const storedProfiles = readStoredProfiles();
const lastProfileId = readLastProfileId();
const initialProfileId = lastProfileId || storedProfiles[0]?.id || DEFAULT_USER_ID;
const initialLearnerProfiles = upsertProfile(
  storedProfiles,
  initialProfileId,
  initialProfileId === DEFAULT_USER_ID ? 'Demo profil' : initialProfileId,
  Boolean(lastProfileId)
);

export const useFocusStore = create<FocusState>((set, get) => ({
  userId: initialProfileId,
  learnerProfiles: initialLearnerProfiles,
  userProfile: null,
  topic: '',
  sessionId: null,
  cameraPermission: 'idle',
  cameraStreamActive: false,
  cameraStatus: null,
  currentState: 'unknown',
  messages: [defaultAiMessage()],
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
  setCameraPermission: (cameraPermission) => set({ cameraPermission }),
  setCameraStreamActive: (cameraStreamActive) => set({ cameraStreamActive }),
  clearCameraStatus: () => set({ cameraStatus: null }),
  selectLearnerProfile: async (profileId) => {
    const normalizedId = normalizeProfileId(profileId);
    if (!normalizedId) {
      set({ error: 'Gecerli bir profil sec.' });
      return;
    }

    const existing = get().learnerProfiles.find((item) => item.id === normalizedId);
    const nextProfiles = upsertProfile(
      get().learnerProfiles,
      normalizedId,
      existing?.label || profileId,
      true
    );

    set({
      userId: normalizedId,
      learnerProfiles: nextProfiles,
      userProfile: null,
      topic: '',
      sessionId: null,
      cameraStatus: null,
      currentState: 'unknown',
      messages: [defaultAiMessage()],
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
      error: null,
    });

    await get().hydrateUserWorkspace(normalizedId);
  },
  createLearnerProfile: async (label) => {
    const trimmedLabel = label.trim();
    const normalizedId = normalizeProfileId(trimmedLabel);

    if (!trimmedLabel || !normalizedId) {
      set({ error: 'Yeni profil icin bir isim gir.' });
      return;
    }

    const nextProfiles = upsertProfile(
      get().learnerProfiles,
      normalizedId,
      trimmedLabel,
      true
    );

    set({
      learnerProfiles: nextProfiles,
      error: null,
    });

    await get().selectLearnerProfile(normalizedId);
  },
  resumeLastProfile: async () => {
    const targetProfileId = readLastProfileId();
    if (!targetProfileId) return;
    await get().selectLearnerProfile(targetProfileId);
  },
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
    const { userId, topic, welcomeData, learnerProfiles } = get();
    const trimmedUserId = normalizeProfileId(userId);

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
      cameraStatus: null,
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
      const nextProfiles = upsertProfile(
        learnerProfiles,
        data.user_id,
        learnerProfiles.find((item) => item.id === data.user_id)?.label || data.user_id,
        true
      );

      set({
        sessionId: data.session_id,
        userId: data.user_id,
        learnerProfiles: nextProfiles,
        topic: data.topic ?? '',
        cameraStatus: null,
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
        get().fetchWelcome(data.user_id, data.topic ?? (topic.trim() || fallbackTopic || '')),
        get().fetchUserProfile(data.user_id),
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
          cameraStatus: null,
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
        mentorMode: data.response_policy ?? null,
        mentorReasons: data.response_reasons ?? [],
        dominantSignals: data.dominant_signals ?? [],
        policyPath: data.policy_path ?? [],
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
          cameraStatus: null,
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
        cameraStatus: null,
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
    const activeUserId = normalizeProfileId(targetUserId ?? userId);

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
    const activeUserId = normalizeProfileId(targetUserId ?? userId);

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
    const activeUserId = normalizeProfileId(targetUserId ?? userId);

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

  fetchWelcome: async (targetUserId, topicOverride) => {
    const { userId, topic } = get();
    const activeUserId = normalizeProfileId(targetUserId ?? userId);
    const resolvedTopic = (topicOverride ?? topic).trim();

    if (!activeUserId) {
      set({ welcomeData: null });
      return;
    }

    try {
      const query = resolvedTopic
        ? `?topic=${encodeURIComponent(resolvedTopic)}`
        : '';
      const res = await fetch(`${API_BASE}/welcome/${encodeURIComponent(activeUserId)}${query}`);
      if (!res.ok) return;

      const data: WelcomeResponse = await res.json();
      set({ welcomeData: data });
    } catch {
      // yardimci veri
    }
  },

  fetchUserProfile: async (targetUserId) => {
    const { userId } = get();
    const activeUserId = normalizeProfileId(targetUserId ?? userId);

    if (!activeUserId) {
      set({ userProfile: null });
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/dashboard/profile/${encodeURIComponent(activeUserId)}`
      );
      if (!res.ok) return;

      const data: UserProfileResponse = await res.json();
      set({ userProfile: data });
    } catch {
      // yardimci veri
    }
  },

  hydrateUserWorkspace: async (targetUserId) => {
    const { userId } = get();
    const activeUserId = normalizeProfileId(targetUserId ?? userId);

    if (!activeUserId) {
      set({
        userProfile: null,
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
      get().fetchUserProfile(activeUserId),
    ]);
  },

  selectHistorySession: async (targetSessionId) => {
    await get().fetchSessionMessages(targetSessionId);
  },

  pushCameraFrame: async (imageBase64) => {
    const { sessionId, userId } = get();

    if (!sessionId || !imageBase64) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/camera/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
          image_base64: imageBase64,
        }),
      });

      if (res.status === 404) {
        set({ cameraStatus: null });
        return;
      }

      if (!res.ok) {
        throw new Error(await parseError(res));
      }

      const data: CameraStatusResponse = await res.json();
      set({ cameraStatus: data });
    } catch (err) {
      set((state) => ({
        cameraStatus: {
          session_id: sessionId,
          status: 'error',
          available: false,
          active: state.cameraStreamActive,
          face_detected: false,
          frame_id: state.cameraStatus?.frame_id ?? 0,
          error: err instanceof Error ? err.message : 'Kamera frame gonderilemedi.',
        },
      }));
    }
  },

  submitFeedback: async ({
    feedbackType,
    interventionType,
    sessionId,
    messageId,
    notes,
  }) => {
    const { userId } = get();
    const activeUserId = normalizeProfileId(userId);

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
