/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        intro: { 50: "#EFF6FF", 100: "#DBEAFE", 500: "#3B82F6", 700: "#1D4ED8" },
        back:  { 50: "#ECFEFF", 100: "#CFFAFE", 500: "#06B6D4", 700: "#0E7490" },
        meth:  { 50: "#ECFDF5", 100: "#D1FAE5", 500: "#10B981", 700: "#047857" },
        res:   { 50: "#FFFBEB", 100: "#FEF3C7", 500: "#F59E0B", 700: "#B45309" },
        disc:  { 50: "#FEF2F2", 100: "#FEE2E2", 500: "#EF4444", 700: "#B91C1C" },
        conc:  { 50: "#F5F3FF", 100: "#EDE9FE", 500: "#8B5CF6", 700: "#6D28D9" },
        contr: { 50: "#FDF2F8", 100: "#FCE7F3", 500: "#EC4899", 700: "#BE185D" },
        lim:   { 50: "#F9FAFB", 100: "#F3F4F6", 500: "#6B7280", 700: "#374151" },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['Georgia', 'Cambria', 'serif'],
      },
    },
  },
  plugins: [],
};
