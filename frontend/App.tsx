import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useFocusStore } from './store';

export default function App() {
  const { scores, addScore } = useFocusStore();
  const [streak, setStreak] = useState(12); // Örnek: 12 dakikadır odaklı
  const [goalProgress, setGoalProgress] = useState(65); // Örnek: Hedefin %65'i tamam

  useEffect(() => {
    const interval = setInterval(() => {
      addScore(Math.floor(Math.random() * 25) + 70);
    }, 2000);
    return () => clearInterval(interval);
  }, [addScore]);

  // Tasarım Ayarları
  const containerStyle: React.CSSProperties = { backgroundColor: '#f8fafc', minHeight: '100vh', padding: '40px', fontFamily: 'sans-serif', color: '#1e293b' };
  const cardStyle: React.CSSProperties = { backgroundColor: 'white', borderRadius: '30px', padding: '25px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', border: '1px solid #e2e8f0' };
  const accentCardStyle: React.CSSProperties = { background: 'linear-gradient(135deg, #4f46e5 0%, #2563eb 100%)', borderRadius: '30px', padding: '25px', color: 'white', boxShadow: '0 10px 15px -3px rgba(37, 99, 235, 0.2)' };

  return (
    <div style={containerStyle}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{ backgroundColor: '#2563eb', padding: '10px', borderRadius: '12px', color: 'white' }}>🧠</div>
          <h1 style={{ fontSize: '24px', fontWeight: '900', margin: 0 }}>FocusAI</h1>
        </div>
        {/* Odak Serisi Ateşi */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#fff', padding: '10px 20px', borderRadius: '50px', boxShadow: '0 2px 5px rgba(0,0,0,0.05)', border: '1px solid #e2e8f0' }}>
          <span style={{ fontSize: '20px' }}>🔥</span>
          <span style={{ fontWeight: 'bold', color: '#f59e0b' }}>{streak} Dakika Odak Serisi!</span>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px' }}>
        
        {/* SOL KOLON */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
          {/* Grafik */}
          <div style={cardStyle}>
            <h2 style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '20px' }}>📈 Odak Yolculuğum</h2>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scores}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} stroke="#cbd5e1" fontSize={12} />
                  <Tooltip contentStyle={{ borderRadius: '15px', border: 'none' }} />
                  <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={5} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Günün Hedefi Progress Bar */}
          <div style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ fontWeight: 'bold', fontSize: '14px' }}>Günün Hedefi</span>
              <span style={{ color: '#2563eb', fontWeight: 'bold' }}>%{goalProgress}</span>
            </div>
            <div style={{ width: '100%', height: '12px', backgroundColor: '#f1f5f9', borderRadius: '10px', overflow: 'hidden' }}>
              <div style={{ width: `${goalProgress}%`, height: '100%', backgroundColor: '#2563eb', transition: 'width 0.5s ease-in-out' }}></div>
            </div>
            <p style={{ fontSize: '12px', color: '#64748b', marginTop: '10px' }}>Bugünkü 2 saatlik odaklanma hedefine çok yaklaştın!</p>
          </div>
        </div>

        {/* SAĞ KOLON */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
          <div style={accentCardStyle}>
            <h3 style={{ fontSize: '12px', fontWeight: 'bold', opacity: 0.8, textTransform: 'uppercase', marginBottom: '15px' }}>✨ Senin Odak Koçun</h3>
            <p style={{ fontSize: '18px', fontWeight: '500', lineHeight: '1.5' }}>
              "Müthiş gidiyorsun! Son 10 dakikadır gözlerini ekrandan ayırmadın. Böyle devam edersen bugün 'Odak Şampiyonu' rozetini alabilirsin!"
            </p>
          </div>

          <div style={{ ...cardStyle, textAlign: 'center' }}>
            <div style={{ fontSize: '48px', fontWeight: '900', color: '#2563eb' }}>
              %{scores.length > 0 ? scores[scores.length - 1].value : '--'}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 'bold' }}>ANLIK ODAK PUANIN</div>
          </div>

          {/* Yeni Rozet Kutusu */}
          <div style={{ ...cardStyle, textAlign: 'center', backgroundColor: '#f0f9ff', borderColor: '#bae6fd' }}>
            <div style={{ fontSize: '30px' }}>🏅</div>
            <div style={{ fontWeight: 'bold', color: '#0369a1', fontSize: '14px' }}>Kazanılan Rozetler</div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '10px' }}>
              <span title="Erken Kuş">🌅</span>
              <span title="Derin Odak">🌊</span>
              <span title="Haftalık Seri">💎</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}