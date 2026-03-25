import React, { useEffect, useState } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { io } from 'socket.io-client';
import logo from './logo.jpeg'; 

const socket = io('http://localhost:5000');

export default function App() {
  const [scores, setScores] = useState<{ value: number; time: string }[]>([]);
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Selam! Bugün hangi konuyu çalışıyoruz? Sana ders notlarından yardımcı olabilirim.' }
  ]);
  const [input, setInput] = useState('');
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraFrame, setCameraFrame] = useState('');
  const [stats, setStats] = useState({ totalTime: "0 dk", avgSuccess: "%0" });

  // --- HAFTALIK MOTİVASYON SÖZLERİ ---
  const dailyQuotes = [
    "Pazartesi: Başlamak için mükemmel olmana gerek yok, ama mükemmel olmak için başlaman gerek.",
    "Salı: Bugün yapacağın küçük bir çalışma, yarınki büyük başarın için bir adımdır.",
    "Çarşamba: Zihnini odakla, sınırlarını zorla. Başarı odaklandığın yerdedir.",
    "Perşembe: Zorluklar, başarıyı daha tatlı kılan basamaklardır. Devam et!",
    "Cuma: Haftayı güçlü bitir! Unutma; disiplin, hedeflerle başarı arasındaki köprüdür.",
    "Cumartesi: Bugün kendine bir yatırım yap ve öğrenmeye devam et.",
    "Pazar: Yarının kazananı, bugün vazgeçmeyen kişidir."
  ];

  // Sistemin hangi günde olduğunu anlayan fonksiyon (0=Pazar, 1=Pazartesi...)
  const dayIndex = new Date().getDay();
  // Pazar 0 olduğu için diziyi ona göre hizalamak gerekirse (Pazartesi 1, Salı 2... Pazar 0)
  // Dizimizde Pazartesi 0. indexte olduğu için küçük bir ayar yapıyoruz:
  const quoteIndex = dayIndex === 0 ? 6 : dayIndex - 1;
  const todayQuote = dailyQuotes[quoteIndex];

  useEffect(() => {
    socket.on('focus_data', (data: { value: number }) => {
      setScores(prev => [...prev.slice(-15), { value: data.value, time: '' }]);
    });
    socket.on('ai_status', (data: { totalTime: string, avgSuccess: string }) => {
      setStats({ totalTime: data.totalTime, avgSuccess: data.avgSuccess });
    });
    socket.on('ai_response', (data: { text: string }) => {
      setMessages(prev => [...prev, { role: 'ai', text: data.text }]);
    });
    socket.on('video_frame', (data: { image: string }) => {
      setCameraFrame(data.image);
    });
    return () => {
      socket.off('focus_data');
      socket.off('ai_status');
      socket.off('ai_response');
      socket.off('video_frame');
    };
  }, []);

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages(prev => [...prev, { role: 'user', text: input }]);
    socket.emit('ask_ai', { question: input });
    setInput('');
  };

  return (
    <div style={{ backgroundColor: '#f0f4f8', minHeight: '100vh', padding: '30px', fontFamily: 'Inter, sans-serif', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* ÜST BAR */}
      <header style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <img src={logo} alt="FocusAI Logo" style={{ width: '45px', height: '45px', borderRadius: '10px', objectFit: 'cover' }} />
        <h1 style={{ color: '#1e293b', fontWeight: 800, margin: 0, fontSize: '24px' }}>
          FocusAI <span style={{color: '#3b82f6'}}>Assistant</span>
        </h1>
        <button 
          onClick={() => setIsCameraOpen(!isCameraOpen)}
          style={{ 
            backgroundColor: isCameraOpen ? '#ef4444' : '#ffffff', 
            color: isCameraOpen ? 'white' : '#1e293b', 
            border: '1px solid #e2e8f0', padding: '8px 16px', borderRadius: '10px', cursor: 'pointer', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)', marginLeft: '10px'
          }}
        >
          {isCameraOpen ? '✕ Kapat' : '📷 Camera Analysis'}
        </button>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '25px', flex: 1 }}>
        
        {/* SOL: CHATBOX */}
        <div style={{ backgroundColor: 'white', borderRadius: '24px', display: 'flex', flexDirection: 'column', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)', overflow: 'hidden' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid #f1f5f9', background: '#f8fafc' }}>
            <h3 style={{ margin: 0, fontSize: '16px' }}>🤖 AI Chatbox (RAG)</h3>
          </div>
          <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {messages.map((m, i) => (
              <div key={i} style={{ 
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                backgroundColor: m.role === 'user' ? '#1e40af' : '#f1f5f9',
                color: m.role === 'user' ? 'white' : '#1e293b',
                padding: '12px 18px', borderRadius: '18px', maxWidth: '75%', fontSize: '14px'
              }}>
                <strong>{m.role === 'ai' ? 'ai: ' : 'user: '}</strong>{m.text}
              </div>
            ))}
          </div>
          <div style={{ padding: '20px', display: 'flex', gap: '10px', background: 'white', borderTop: '1px solid #f1f5f9' }}>
            <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()} placeholder="Mesajınızı yazın..." style={{ flex: 1, padding: '12px 20px', borderRadius: '12px', border: '1px solid #cbd5e1', outline: 'none' }} />
            <button onClick={handleSend} style={{ backgroundColor: '#1e40af', color: 'white', border: 'none', padding: '0 25px', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold' }}>Gönder</button>
          </div>
        </div>

        {/* SAĞ: PANELLER */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          
          {/* GÜNÜN MOTİVASYONU (Haftalık Değişir) */}
          <div style={{ backgroundColor: '#1e40af', padding: '20px', borderRadius: '24px', color: 'white', boxShadow: '0 4px 15px rgba(30, 64, 175, 0.2)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
              <span style={{ fontSize: '20px' }}>⭐</span>
              <h4 style={{ margin: 0, fontSize: '13px', letterSpacing: '0.5px' }}>GÜNÜN MOTİVASYON SÖZÜ</h4>
            </div>
            <p style={{ fontSize: '13.5px', fontStyle: 'italic', lineHeight: '1.5', margin: 0, opacity: 0.95 }}>
              "{todayQuote}"
            </p>
          </div>

          {/* GÜNLÜK VERİMLİLİK */}
          <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '24px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '13px', color: '#64748b' }}>GÜNLÜK VERİMLİLİK</h4>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#1e293b' }}>{stats.totalTime}</div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Toplam Odak</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#10b981' }}>{stats.avgSuccess}</div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Ort. Başarı</div>
              </div>
            </div>
          </div>

          {/* KÜÇÜLTÜLMÜŞ GRAFİK */}
          <div style={{ backgroundColor: 'white', padding: '15px', borderRadius: '24px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', height: '180px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <h4 style={{ margin: 0, fontSize: '12px', color: '#64748b' }}>CANLI ODAK ANALİZİ</h4>
              <span style={{ color: '#3b82f6', fontWeight: 'bold', fontSize: '12px' }}>%{scores.length > 0 ? scores[scores.length-1].value : 0}</span>
            </div>
            <div style={{ flex: 1 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scores}>
                  <YAxis domain={[0, 100]} hide />
                  <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={3} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </div>

      {/* KAMERA PENCERESİ */}
      {isCameraOpen && (
        <div style={{
          position: 'fixed', bottom: '30px', right: '30px', width: '300px', backgroundColor: '#1e293b', borderRadius: '20px', padding: '10px', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)', zIndex: 1000, border: '2px solid #3b82f6'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'white', marginBottom: '8px', fontSize: '12px' }}>
            <span>🔵 Canlı AI Analiz</span>
            <button onClick={() => setIsCameraOpen(false)} style={{ color: 'white', background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
          </div>
          <img src={cameraFrame ? `data:image/jpeg;base64,${cameraFrame}` : "https://via.placeholder.com/300x220/1e293b/ffffff?text=Kamera+Bekleniyor..."} style={{ width: '100%', borderRadius: '12px' }} alt="AI Stream" />
        </div>
      )}
    </div>
  );
}