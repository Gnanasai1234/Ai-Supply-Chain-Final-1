import { Canvas, useFrame } from '@react-three/fiber'
import { useNavigate } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Boxes, CircleCheck, Eye, EyeOff, Network } from 'lucide-react'
import api from './api'

const points = [[-2.8, 1.3, 0], [-1.5, -.2, .3], [-.2, .8, -.2], [1.15, -.3, .25], [2.8, .65, 0]]
function Nodes({ reduceMotion }) {
  const group = useRef()
  useFrame(({ clock }) => { if (!reduceMotion && group.current) group.current.rotation.y = Math.sin(clock.elapsedTime * .2) * .08 })
  return <group ref={group}>{points.map((p, i) => <group key={i} position={p}>
    <mesh><sphereGeometry args={[i === 2 ? .16 : .11, 24, 24]} /><meshStandardMaterial color={i === 2 ? '#82d5b6' : '#8b9bfa'} emissive={i === 2 ? '#1c7057' : '#303d92'} emissiveIntensity={.55} /></mesh>
    <mesh position={[0, 0, -.03]}><sphereGeometry args={[.25, 18, 18]} /><meshBasicMaterial color="#9aabff" transparent opacity={.11} /></mesh>
  </group>)}</group>
}
function Links() {
  return <group>{points.slice(0, -1).map((p, i) => {
    const q = points[i + 1]
    const mid = p.map((value, axis) => (value + q[axis]) / 2)
    const length = Math.hypot(...p.map((value, axis) => value - q[axis]))
    return <mesh key={i} position={mid} rotation={[0, 0, Math.atan2(q[1] - p[1], q[0] - p[0])]}><cylinderGeometry args={[.009, .009, length, 8]} /><meshBasicMaterial color="#566be1" transparent opacity={.56} /></mesh>
  })}</group>
}

export default function Login() {
  const navigate = useNavigate()
  const [reduceMotion, setReduceMotion] = useState(false)
  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduceMotion(preference.matches)
    update()
    preference.addEventListener?.('change', update)
    return () => preference.removeEventListener?.('change', update)
  }, [])
  useEffect(() => {
    if (!sessionStorage.getItem('chainwise_access_token')) return
    api.get('/auth/me').then(() => navigate('/dashboard', { replace: true })).catch(() => sessionStorage.removeItem('chainwise_access_token'))
  }, [navigate])
  const submit = async event => {
    event.preventDefault()
    setError('')
    if (mode === 'register' && password !== confirm) { setError('The passwords do not match.'); return }
    setBusy(true)
    try {
      const endpoint = mode === 'register' ? '/auth/register' : '/auth/login'
      const payload = mode === 'register' ? { name, email, password } : { email, password }
      const { data } = await api.post(endpoint, payload)
      sessionStorage.setItem('chainwise_access_token', data.access_token)
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not sign in. Check the API connection and try again.')
    } finally { setBusy(false) }
  }
  return <main className="login-shell"><section className="login-left"><div className="brand"><span className="brand-icon"><Boxes size={21} /></span> chainwise</div><div className="login-copy"><span className="eyebrow">SUPPLY CHAIN INTELLIGENCE</span><h1>See the signal.<br /><em>Stay ahead.</em></h1><p>A clearer view of disruption risk, from the first supplier signal to the last delivery mile.</p><div className="login-points"><span><CircleCheck size={16} /> Early risk visibility</span><span><CircleCheck size={16} /> Actionable response plans</span></div>
    <form className="auth-form" onSubmit={submit}>
      {mode === 'register' && <label>Name<input required autoComplete="name" value={name} onChange={event => setName(event.target.value)} /></label>}
      <label>Email<input required type="email" autoComplete="email" value={email} onChange={event => setEmail(event.target.value)} /></label>
      <label>
        Password
        <div className="password-input-wrapper">
          <input
            required
            minLength={8}
            type={showPassword ? 'text' : 'password'}
            autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
            value={password}
            onChange={event => setPassword(event.target.value)}
          />
          <button
            type="button"
            className="password-toggle-btn"
            onClick={() => setShowPassword(v => !v)}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            title={showPassword ? 'Hide password' : 'Show password'}
            tabIndex={-1}
          >
            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
      </label>
      {mode === 'register' && (
        <label>
          Confirm password
          <div className="password-input-wrapper">
            <input
              required
              minLength={8}
              type={showConfirm ? 'text' : 'password'}
              autoComplete="new-password"
              value={confirm}
              onChange={event => setConfirm(event.target.value)}
            />
            <button
              type="button"
              className="password-toggle-btn"
              onClick={() => setShowConfirm(v => !v)}
              aria-label={showConfirm ? 'Hide confirm password' : 'Show confirm password'}
              title={showConfirm ? 'Hide confirm password' : 'Show confirm password'}
              tabIndex={-1}
            >
              {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </label>
      )}
      {error && <p role="alert" className="auth-error">{error}</p>}
      <button className="primary-btn login-btn" disabled={busy}>{busy ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Sign in'} <ArrowRight size={17} /></button>
    </form>
    <button className="auth-switch" type="button" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(''); setShowPassword(false); setShowConfirm(false) }}>{mode === 'login' ? 'New to Chainwise? Create an account' : 'Already have an account? Sign in'}</button>
    </div><footer>Decision support for resilient supply chains</footer></section><section className="login-visual"><div className="visual-glow" /><div className="network-label"><Network size={15} /> SUPPLY NETWORK MODEL</div><Canvas frameloop={reduceMotion ? 'demand' : 'always'} camera={{ position: [0, 0, 7], fov: 55 }}><ambientLight intensity={1.2} /><pointLight position={[3, 3, 4]} intensity={2} color="#90a3ff" /><Nodes reduceMotion={reduceMotion} /><Links /></Canvas><div className="stages">{['Supplier', 'Factory', 'Warehouse', 'Distributor', 'Customer'].map((s, i) => <span key={s}><i className={i === 2 ? 'stage-dot active' : 'stage-dot'} />{s}</span>)}</div><div className="visual-caption"><b>One connected picture</b><span>Signals across your supply network, in context.</span></div><div className="visual-index">01 — NETWORK OVERVIEW</div></section></main>
}
