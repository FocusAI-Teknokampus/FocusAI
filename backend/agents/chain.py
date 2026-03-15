# backend/agents/chain.py
from backend.agents.action_agent import ActionAgent

class AgentCoordinator:
    def __init__(self):
        # Ajanlarımızı başlatıyoruz
        self.action_agent = ActionAgent()
        # İleride: self.context_agent = ContextAgent() vb.

    def process_low_attention(self, score: float, reasons: list[str]) -> dict:
        """
        Düşük dikkat skoru geldiğinde hangi ajanların çalışacağına karar verir.
        (LangGraph Node yapısı ileride buraya entegre edilecek)
        """
        # Şimdilik doğrudan Action Agent'a gidip aksiyon önerisi alıyoruz
        advice = self.action_agent.get_advice(score, reasons)
        
        return {
            "status": "warning",
            "action_advice": advice
        }