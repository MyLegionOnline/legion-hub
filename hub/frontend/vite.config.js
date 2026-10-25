import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: parseInt(process.env.PORT || '3000'),
    host: '0.0.0.0',
    // Poll for file changes — needed on Docker Desktop / Windows volume mounts
    watch: {
      usePolling: true,
      interval: 500,
    },
    proxy: {
      '/api': {
        // Inside Docker the backend is reachable via service name
        target: process.env.VITE_BACKEND_URL || 'http://backend:8000',
        changeOrigin: true,
        ws: true,
      }
    }
  }
})
