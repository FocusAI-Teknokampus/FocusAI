import { create } from 'zustand';

interface FocusState {
  scores: { time: string; value: number }[];
  addScore: (score: number) => void;
}

export const useFocusStore = create<FocusState>((set) => ({
  scores: [],
  addScore: (score) => set((state) => ({
    scores: [...state.scores, { 
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), 
      value: score 
    }].slice(-15)
  })),
}));