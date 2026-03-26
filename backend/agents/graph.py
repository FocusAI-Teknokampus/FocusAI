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

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

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

from backend.state.feature_extractor import FeatureExtractor
from backend.state.state_model import StateModel
from backend.state.uncertainty_engine import UncertaintyEngine
from backend.agents.mentor_agent import MentorAgent


# ─────────────────────────────────────────────────────────────────────
# Tekil instance'lar
# ─────────────────────────────────────────────────────────────────────
# Neden?
#   FeatureExtractor session bazlı sayaç tutuyor.
#   UncertaintyEngine session bazlı cooldown tutuyor.
#   Bunlar her request'te yeniden yaratılırsa hafızaları sıfırlanır.

_feature_extractor = FeatureExtractor()
_state_model = StateModel()
_uncertainty_engine = UncertaintyEngine()
_mentor_agent = MentorAgent()


# ─────────────────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────────────────

class MentorGraphState(TypedDict):
    message: ChatMessage

    session_context: Optional[ShortTermContext]
    user_profile: Optional[UserProfile]
    baseline_profile: Optional[dict]

    feature_vector: Optional[FeatureVector]
    state_estimate: Optional[StateEstimate]

    should_intervene: bool
    intervention: Optional[MentorIntervention]
    rag_context: Optional[str]
    rag_source: Optional[str]

    llm_response: Optional[str]
    final_response: Optional[ChatResponse]

    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────
# NODE'LAR
# ─────────────────────────────────────────────────────────────────────

def session_node(state: MentorGraphState) -> dict:
    """
    Oturum bağlamını ve kullanıcı profilini yükler.

    Burada:
        - aktif session RAM'den okunur
        - kullanıcı profili long-term memory'den alınır
    """
    from backend.agents.session_agent import SessionAgent
    from backend.core.database import SessionLocal
    from backend.services.baseline_service import BaselineService

    agent = SessionAgent()
    message = state["message"]

    context = agent.load_context(message.session_id)
    profile = agent.load_profile(message.user_id)
    baseline_profile = None

    db = SessionLocal()
    try:
        baseline_profile = BaselineService(db).get_state_model_baseline(message.user_id)
    finally:
        db.close()

    return {
        "session_context": context,
        "user_profile": profile,
        "baseline_profile": baseline_profile,
    }


def feature_node(state: MentorGraphState) -> dict:
    """
    Ham kullanıcı mesajından davranış sinyalleri çıkarır.

    Önemli:
        FeatureExtractor tekil instance'tır.
        Böylece retry_count, idle_time ve topic geçmişi sıfırlanmaz.
    """
    msg = state["message"]

    feature = _feature_extractor.extract(
        session_id=msg.session_id,
        message_content=msg.content,
        message_timestamp=msg.timestamp,
        channel=msg.channel,
        camera_signal=None,   # Kamera sinyali daha sonra bağlanabilir
    )

    return {"feature_vector": feature}


def state_node(state: MentorGraphState) -> dict:
    """
    FeatureVector'dan kullanıcı state tahmini üretir.
    """
    profile = state.get("user_profile")
    threshold = profile.adaptive_threshold if profile else None

    estimate = _state_model.predict(
        features=state["feature_vector"],
        adaptive_threshold=threshold,
        baseline_profile=state.get("baseline_profile"),
    )

    return {"state_estimate": estimate}


def uncertainty_node(state: MentorGraphState) -> dict:
    """
    Müdahale gerekip gerekmediğine karar verir.

    Önemli:
        UncertaintyEngine tekil instance'tır.
        Böylece session bazlı cooldown mekanizması çalışır.
    """
    intervention = _uncertainty_engine.decide(
        estimate=state["state_estimate"],
        profile=state.get("user_profile"),
        session_id=state["message"].session_id,
        policy_summary=_load_policy_summary(
            user_id=state["message"].user_id,
            estimate=state["state_estimate"],
        ),
    )

    return {
        "should_intervene": intervention is not None,
        "intervention": intervention,
    }


def _load_policy_summary(user_id: str, estimate: Optional[StateEstimate]) -> dict:
    if estimate is None or estimate.state in [UserState.UNKNOWN, UserState.FOCUSED]:
        return {}

    from backend.core.database import SessionLocal
    from backend.services.intervention_policy_service import InterventionPolicyService

    db = SessionLocal()
    try:
        service = InterventionPolicyService(db)
        return service.get_state_policy_summary(
            user_id=user_id,
            state_label=estimate.state.value,
        )
    finally:
        db.close()


def mentor_node(state: MentorGraphState) -> dict:
    """
    Müdahale gerekiyorsa LLM ile müdahale metnini kişiselleştirir.
    """
    intervention = state.get("intervention")
    if intervention is None:
        return {"intervention": None}

    enriched = _mentor_agent.enrich_intervention(
        intervention=intervention,
        user_profile=state.get("user_profile"),
        session_context=state.get("session_context"),
    )

    return {"intervention": enriched}


def clarify_node(state: MentorGraphState) -> dict:
    """
    Confidence düşükse doğrudan müdahale etmek yerine açıklayıcı soru sorar.
    """
    estimate = state.get("state_estimate")
    if not estimate:
        return {}

    question_map = {
        UserState.STUCK: "Takıldığın bir yer var gibi görünüyor — biraz daha anlatır mısın?",
        UserState.FATIGUED: "Biraz yorgun görünüyorsun. Kısa bir mola ister misin?",
        UserState.DISTRACTED: "Dikkatin dağılmış olabilir. Devam etmek mi istersin, kısa bir toparlama mı yapalım?",
    }

    msg = question_map.get(estimate.state)
    if msg is None:
        return {}

    clarifying = MentorIntervention(
        intervention_type=InterventionType.QUESTION,
        message=msg,
        triggered_by=estimate.state,
        confidence=estimate.confidence,
        decision_reason=(
            f"Confidence ({estimate.confidence}) kullanici esiginin ({estimate.threshold}) altinda kaldigi icin "
            "dogrudan mudahale yerine aciklayici soru secildi."
        ),
    )

    return {
        "intervention": clarifying,
        "should_intervene": True,
    }


def rag_node(state: MentorGraphState) -> dict:
    """
    Kullanıcının yüklediği notlarda ilgili içerik arar.
    """
    from backend.rag.rag_agent import RAGAgent

    agent = RAGAgent()
    message = state["message"]

    if not agent.has_notes(message.user_id):
        return {
            "rag_context": None,
            "rag_source": None,
        }

    result = agent.search(
        user_id=message.user_id,
        query=message.content,
    )

    # RAG bulunduysa hem LLM'e gidecek metni hem de UI'da gosterilecek kaynagi tasiyoruz.
    # Boylece backend response modeli ile frontend gorunumu ayni veriye dayanir.
    return {
        "rag_context": result.source_chunk if result.found else None,
        "rag_source": result.filename if result.found else None,
    }


def response_node(state: MentorGraphState) -> dict:
    """
    Tüm bağlamı kullanarak final mentor yanıtını üretir.
    """
    response_text = _mentor_agent.generate_response(
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
        # Not kaynagi varsa API response'a ekliyoruz.
        # Frontend bunu "hangi nottan beslendi" bilgisi olarak gosterebilir.
        rag_source=state.get("rag_source"),
        mentor_intervention=state.get("intervention"),
        current_state=estimate.state if estimate else UserState.UNKNOWN,
    )

    return {
        "llm_response": response_text,
        "final_response": final,
    }


# ─────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────

def route_after_uncertainty(state: MentorGraphState) -> str:
    """
    uncertainty_node sonrasında hangi path'in izleneceğine karar verir.
    """
    intervention = state.get("intervention")
    estimate = state.get("state_estimate")

    if intervention is None:
        return "rag_node"

    # Confidence eşiği kullanıcıya özel threshold ile hizalı olsun.
    if estimate and estimate.confidence >= estimate.threshold:
        return "mentor_node"

    # Confidence düşükse açıklayıcı soru
    return "clarify_node"


# ─────────────────────────────────────────────────────────────────────
# GRAPH TANIMI
# ─────────────────────────────────────────────────────────────────────

graph = StateGraph(MentorGraphState)

graph.add_node("session_node", session_node)
graph.add_node("feature_node", feature_node)
graph.add_node("state_node", state_node)
graph.add_node("uncertainty_node", uncertainty_node)
graph.add_node("mentor_node", mentor_node)
graph.add_node("clarify_node", clarify_node)
graph.add_node("rag_node", rag_node)
graph.add_node("response_node", response_node)

graph.set_entry_point("session_node")

graph.add_edge("session_node", "feature_node")
graph.add_edge("feature_node", "state_node")
graph.add_edge("state_node", "uncertainty_node")

graph.add_conditional_edges(
    "uncertainty_node",
    route_after_uncertainty,
    {
        "mentor_node": "mentor_node",
        "clarify_node": "clarify_node",
        "rag_node": "rag_node",
    },
)

graph.add_edge("mentor_node", "rag_node")
graph.add_edge("clarify_node", "rag_node")
graph.add_edge("rag_node", "response_node")
graph.add_edge("response_node", END)

mentor_graph = graph.compile()
