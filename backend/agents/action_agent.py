# backend/agents/action_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from backend.core.config import settings
import os

class ActionAgent:
    def __init__(self):
        # LangChain'in OpenAI entegrasyonunu başlatıyoruz
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key, 
            model="gpt-4o-mini",
            temperature=0.5
        )
        
        # Jinja2 şablonumuzu okuyoruz
        current_dir = os.path.dirname(__file__)
        template_path = os.path.join(current_dir, "prompts", "action.j2")
        
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # LangChain PromptTemplate oluşturuyoruz (Jinja2 formatında)
        self.prompt = PromptTemplate.from_template(
            template_content, 
            template_format="jinja2"
        )

    def get_advice(self, score: float, reasons: list[str]) -> str:
        reasons_text = ", ".join(reasons) if reasons else "Bilinmiyor"
        
        # Zinciri (Chain) oluştur ve çalıştır: Prompt'u doldur -> LLM'e gönder
        chain = self.prompt | self.llm
        
        response = chain.invoke({"score": score, "reasons": reasons_text})
        return response.content