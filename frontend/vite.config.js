import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  envDir: '..',
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (id.includes('/three/') || id.includes('/@react-three/')) return 'three'
          if (id.includes('/recharts/') || id.includes('/d3-')) return 'charts'
          if (id.includes('/react/') || id.includes('/react-dom/') || id.includes('/react-router')) return 'react'
        },
      },
    },
  },
})
