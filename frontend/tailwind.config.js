/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.{js,ts,jsx,tsx}", // Bu satır en dıştaki App.tsx ve main.tsx'i kapsar
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}