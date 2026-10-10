import React, { lazy, Suspense, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import api from './api'
import './styles.css'
import './enhancements.css'

const App = lazy(() => import('./App.jsx'))
const Login = lazy(() => import('./Login.jsx'))

function ProtectedDashboard() {
  const [valid, setValid] = useState(null)
  useEffect(() => {
    if (!sessionStorage.getItem('chainwise_access_token')) { setValid(false); return }
    api.get('/auth/me').then(() => setValid(true)).catch(() => {
      sessionStorage.removeItem('chainwise_access_token')
      setValid(false)
    })
  }, [])
  if (valid === null) return <div className="route-loading">Checking your session…</div>
  return valid ? <App /> : <Navigate to="/" replace />
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode><BrowserRouter><Suspense fallback={<div className="route-loading">Loading local workspace…</div>}><Routes><Route path="/" element={<Login />} /><Route path="/dashboard/*" element={<ProtectedDashboard />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></Suspense></BrowserRouter></React.StrictMode>
)
