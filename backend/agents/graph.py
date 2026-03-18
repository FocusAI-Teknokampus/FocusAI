# backend/agents/graph.py
#
# LangGraph state machine — tüm agent'ları birbirine bağlayan iskelet.
#
# Akış:
#   [session_node] → [feature_node] → [state_node] → [uncertainty_node]
#                                                           ↓
#                                              confidence yeterli mi?
#                                             /                      \
#                                      [mentor_node]           [clarify_node]
#                                       müdahale et             soru sor
#                                             \                      /
#                                          [rag_node]
#                                        not var mı ara
#                                             ↓
#                                        [response_node]
#                                        LLM'e gönder, cevapla
#
# Sahip: K1
# Bağımlılıklar: schemas.py, state_model.py, uncertainty_engine.py
#                session_agent.py, mentor_agent.py (aşağıda yazılacak)

from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator

from backend.core.schemas import (
    ChatMessage,
    ChatResponse,
    FeatureVector,
    InterventionType,
    MentorIntervention,
    ShortTermContext,
    StateEstimate,
    UserProfile,
    UserState,
)


# ─────────────────────────────────────────────────────────────────────
# GRAPH STATE — node'lar arası taşınan veri paketi
# Her node bu paketi alır, istediğini günceller, bir sonrakine geçer.
# ─────────────────────────────────────────────────────────────────────

class MentorGraphState(TypedDict):
    # Gelen mesaj
    message: ChatMessage

    # Session ve kullanıcı bağlamı
    session_context: Optional[ShortTermContext]
    user_profile: Optional[UserProfile]

    # Analiz sonuçları
    feature_vector: Optional[FeatureVector]
    state_estimate: Optional[StateEstimate]

    # Karar sonuçları
    should_intervene: bool
    intervention: Optional[MentorIntervention]
    rag_context: Optional[str]          # Nottan bulunan ilgili içerik

    # Final çıktı
    llm_response: Optional[str]
    final_response: Optional[ChatResponse]

    # Hata yönetimi
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────
# NODE FONKSİYONLARI
# Her node state'i alır, günceller, döner.
# İçleri şu an için iskelet — agent'lar yazılınca dolacak.
# ─────────────────────────────────────────────────────────────────────

def session_node(state: MentorGraphState) -> dict:
    """
    Oturum bağlamını yükler.
    Session Agent buraya gelecek (session_agent.py).
    Şimdilik: boş context döner.
    """
    from backend.agents.session_agent import SessionAgent
    agent = SessionAgent()
    context = agent.load_context(state["message"].session_id)
    profile = agent.load_profile(state["message"].user_id)

    return {
        "session_context": context,
        "user_profile": profile,
    }


def feature_node(state: MentorGraphState) -> dict:
    """
    Ham mesajdan sinyal vektörü çıkarır.
    FeatureExtractor (K3) buraya bağlanır.
    Şimdilik: temel davranış sinyalleri.
    """
    from backend.state.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()

    msg = state["message"]
    context = state.get("session_context")

    feature = extractor.extract(
        session_id=msg.session_id,
        message_content=msg.content,
        message_timestamp=msg.timestamp,
        channel=msg.channel,
        camera_signal=None,             # Kamera sinyali Hafta 2'de buraya gelecek
    )

    return {"feature_vector": feature}


def state_node(state: MentorGraphState) -> dict:
    """
    FeatureVector → StateEstimate.
    StateModel burada çalışır.
    """
    from backend.state.state_model import StateModel
    model = StateModel()

    profile = state.get("user_profile")
    threshold = profile.adaptive_threshold if profile else None

    estimate = model.predict(
        features=state["feature_vector"],
        adaptive_threshold=threshold,
    )

    return {"state_estimate": estimate}


def uncertainty_node(state: MentorGraphState) -> dict:
    """
    Müdahale edilmeli mi? Uncertainty Engine karar verir.
    """
    from backend.state.uncertainty_engine import UncertaintyEngine
    engine = UncertaintyEngine()

    intervention = engine.decide(
        estimate=state["state_estimate"],
        profile=state.get("user_profile"),
        session_id=state["message"].session_id,
    )

    return {
        "should_intervene": intervention is not None,
        "intervention": intervention,
    }


def mentor_node(state: MentorGraphState) -> dict:
    """
    Müdahale gerekiyorsa Mentor Agent devreye girer.
    LLM ile kişiselleştirilmiş müdahale mesajı üretir.
    MentorAgent (mentor_agent.py) buraya gelecek.
    """
    from backend.agents.mentor_agent import MentorAgent
    agent = MentorAgent()

    enriched_intervention = agent.enrich_intervention(
        intervention=state["intervention"],
        user_profile=state.get("user_profile"),
        session_context=state.get("session_context"),
    )

    return {"intervention": enriched_intervention}


def clarify_node(state: MentorGraphState) -> dict:
    """
    Confidence düşükse müdahale yerine açıklayıcı soru sor.
    Örnek: "Şu an nasıl hissediyorsun, takıldığın bir yer var mı?"
    """
    estimate = state.get("state_estimate")
    if not estimate:
        return {}

    question_map = {
        UserState.STUCK: "Takıldığın bir yer var gibi görünüyor — biraz daha anlatır mısın?",
        UserState.FATIGUED: "Yorgun görünüyorsun, nasıl hissediyorsun?",
        UserState.DISTRACTED: "Konsantrasyonun dağıldı mı, devam etmek ister misin?",
    }

    msg = question_map.get(estimate.state)
    if msg:
        clarifying = MentorIntervention(
            intervention_type=InterventionType.QUESTION,
            message=msg,
            triggered_by=estimate.state,
            confidence=estimate.confidence,
        )
        return {"intervention": clarifying, "should_intervene": True}

    return {}


def rag_node(state: MentorGraphState) -> dict:
    """
    Kullanıcının notlarında ilgili içerik var mı arar.
    RAG Agent (K2) buraya bağlanacak.
    Şimdilik: None döner (not yok).
    """
    # Hafta 2'de K2'nin RAGAgent'ı buraya gelecek:
    # from backend.agents.rag_agent import RAGAgent
    # agent = RAGAgent()
    # result = agent.search(
    #     query=state["message"].content,
    #     user_id=state["message"].user_id
    # )
    # return {"rag_context": result.source_chunk if result.found else None}

    return {"rag_context": None}


def response_node(state: MentorGraphState) -> dict:
    """
    Tüm bağlamı LLM'e gönderir, final yanıtı üretir.
    Dynamic Persona Prompt burada oluşturulur.
    """
    from backend.agents.mentor_agent import MentorAgent
    agent = MentorAgent()

    response_text = agent.generate_response(
        message=state["message"],
        session_context=state.get("session_context"),
        user_profile=state.get("user_profile"),
        rag_context=state.get("rag_context"),
        intervention=state.get("intervention"),
        state_estimate=state.get("state_estimate"),
    )

    estimate = state.get("state_estimate")

    final = ChatResponse(
        session_id=state["message"].session_id,
        content=response_text,
        rag_source=None,                # RAG gelince burası dolacak
        mentor_intervention=state.get("intervention"),
        current_state=estimate.state if estimate else UserState.UNKNOWN,
    )

    return {
        "llm_response": response_text,
        "final_response": final,
    }


# ─────────────────────────────────────────────────────────────────────
# ROUTING FONKSİYONLARI
# Koşullu edge'ler: "buradan nereye git?"
# ─────────────────────────────────────────────────────────────────────

def route_after_uncertainty(state: MentorGraphState) -> str:
    """
    Uncertainty Engine kararına göre yönlendir.

    Müdahale gerekiyor + confidence yeterli → mentor_node
    Müdahale gerekiyor + confidence düşük   → clarify_node
    Müdahale gerekmez                        → rag_node (direkt cevapla)
    """
    estimate = state.get("state_estimate")
    intervention = state.get("intervention")

    if intervention is None:
        return "rag_node"

    if intervention.intervention_type == InterventionType.QUESTION:
        return "clarify_node"

    return "mentor_node"


# ─────────────────────────────────────────────────────────────────────
# GRAPH TANIMI
# ─────────────────────────────────────────────────────────────────────

def build_mentor_graph() -> StateGraph:
    """
    Graph'ı oluşturur ve döner.
    Her uygulama başlangıcında bir kez çağrılır.

    Kullanım:
        graph = build_mentor_graph()
        app = graph.compile()
        result = app.invoke(initial_state)
    """
    graph = StateGraph(MentorGraphState)

    # Node'ları ekle
    graph.add_node("session_node",     session_node)
    graph.add_node("feature_node",     feature_node)
    graph.add_node("state_node",       state_node)
    graph.add_node("uncertainty_node", uncertainty_node)
    graph.add_node("mentor_node",      mentor_node)
    graph.add_node("clarify_node",     clarify_node)
    graph.add_node("rag_node",         rag_node)
    graph.add_node("response_node",    response_node)

    # Sabit edge'ler (her zaman bu sıra)
    graph.set_entry_point("session_node")
    graph.add_edge("session_node",     "feature_node")
    graph.add_edge("feature_node",     "state_node")
    graph.add_edge("state_node",       "uncertainty_node")

    # Koşullu edge (uncertainty sonrası 3 yol)
    graph.add_conditional_edges(
        "uncertainty_node",
        route_after_uncertainty,
        {
            "mentor_node":   "mentor_node",
            "clarify_node":  "clarify_node",
            "rag_node":      "rag_node",
        }
    )

    # Mentor ve clarify sonrası RAG'a git
    graph.add_edge("mentor_node",  "rag_node")
    graph.add_edge("clarify_node", "rag_node")

    # RAG sonrası her zaman response
    graph.add_edge("rag_node",     "response_node")
    graph.add_edge("response_node", END)

    return graph


# Uygulama genelinde kullanılacak tekil instance
mentor_graph = build_mentor_graph().compile()