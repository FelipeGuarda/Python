import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiBase = process.env.VITE_API_BASE ?? "http://localhost:8000";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api": {
        target: apiBase,
        changeOrigin: true,
      },
      "/ct-images": {
        target: apiBase,
        changeOrigin: true,
      },
    },
  },
})
