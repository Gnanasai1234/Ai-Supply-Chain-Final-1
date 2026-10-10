import { useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { Activity, AlertTriangle, ArrowUpRight, Boxes, Check, ChevronRight, CircleHelp, ClipboardList, Copy, Database, Factory, Gauge, Globe, History as HistoryIcon, Layers, LayoutDashboard, LoaderCircle, LogOut, Menu, Network, Search, ShieldAlert, ShieldCheck, Sparkles, ThumbsDown, ThumbsUp, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import api from './api'
import { AnalysisProgress, ErrorState, HistoricalCaseCard, HistoricalRecordModal, LoadingSkeleton, Toast } from './ui'
import HistoryPage from './History'

const disruptionTypes = ['Factory Incident', 'Cyber Attack', 'Port Congestion', 'Labor Strike', 'Geopolitical', 'Natural Disaster']
const industries = ['Pharmaceuticals', 'Electronics', 'Consumer Goods', 'Automotive', 'Aerospace']
const regions = ['Asia-Pacific', 'Europe', 'South America', 'North America', 'Africa/Middle East']
const sizes = ['Large', 'Medium', 'Small']
const responseTypes = ['Customer Delay', 'Alternative Supplier', 'Production Reroute', 'Inventory Buffer', 'Combined Strategy']
const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase()
const fmt = value => value == null ? 'Not available' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value)
const money = value => value == null ? 'Not available' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)
const friendlyError = err => {
  console.error('Supply-chain API request failed', { status: err.response?.status, code: err.code })
  if (!err.response) return 'The API could not be reached. Check that the backend is running and try again.'
  const detail = err.response.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(item => item.msg).filter(Boolean).join(' ')
  if (err.response.status === 503) return 'The service is unavailable. Check backend health and the required local services.'
  if (err.response.status === 422) return 'Some fields need attention. Review the details and try again.'
  return 'The request could not be completed. Please try again.'
}

const navItems = [
  ['/dashboard', 'Overview', LayoutDashboard], ['/dashboard/analysis', 'Disruption analysis', Sparkles],
  ['/dashboard/history', 'History', HistoryIcon], ['/dashboard/similar-cases', 'Similar cases', Network], ['/dashboard/predictions', 'Predictions', Gauge],
  ['/dashboard/recommendations', 'Recommendations', ClipboardList], ['/dashboard/supplier-risk', 'Supplier risk', ShieldAlert],
  ['/dashboard/resilience', 'Risk & resilience', ShieldAlert],
  ['/dashboard/analytics', 'Analytics', Activity], ['/dashboard/feedback', 'Feedback', ThumbsUp], ['/dashboard/evaluation', 'Evaluation', Database],
]


function Sidebar({ open, close, onLogout, user }) {
  const initials = user?.name ? user.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() : '?'
  return <>{open && <button className="sidebar-scrim" onClick={close} aria-label="Close navigation" />}<aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
    <div className="brand"><span className="brand-icon"><Boxes size={20} /></span><span>Chainwise</span><button className="mobile-close" onClick={close} aria-label="Close navigation"><X size={18} /></button></div>
    <div className="workspace"><div className="workspace-icon">SC</div><div><b>Supply chain dataset</b><small>Local operations workspace</small></div><span className="workspace-pulse" /></div>
    <div className="nav-caption">OPERATIONS</div><nav aria-label="Main navigation">{navItems.map(([to, label, Icon], i) => <NavLink key={to} end={i === 0} to={to} onClick={close}><Icon />{label}<ChevronRight className="nav-chevron" size={14}/></NavLink>)}</nav>
    <div className="sidebar-bottom"><div className="help-card"><span><CircleHelp size={16} /> Evidence first</span><small>Historical records and saved model outputs inform each analysis.</small></div>
      <div className="session-user">
        <span className="session-avatar" aria-hidden="true">{initials}</span>
        <div className="session-info"><b title={user?.name}>{user?.name || 'Loading\u2026'}</b><small title={user?.email}>{user?.email || ''}</small></div>
        <button className="session-logout" onClick={onLogout} title="Log out" aria-label="Log out"><LogOut size={14}/></button>
      </div>
    </div>
  </aside></>
}

function Header({ title, openMenu, health }) {
  const apiReady = health?.status === 'ok'
  const hasModelStatus = !!health?.models
  const availableModels = Object.values(health?.models || {}).filter(Boolean).length
  const modelLabel = !hasModelStatus ? 'Model status unavailable' : availableModels === 4 ? 'Models ready' : `Models ${availableModels}/4`
  return <header className="topbar"><button className="mobile-menu" onClick={openMenu} aria-label="Open navigation"><Menu size={19} /></button><div className="breadcrumbs">Operations <span>/</span> <b>{title}</b></div><div className="top-actions"><span className={`system-status ${apiReady ? '' : 'status-offline'}`}><i />{apiReady ? 'API connected' : 'API unavailable'}</span><span className={`model-status ${hasModelStatus ? '' : 'status-unknown'} ${availableModels === 4 ? 'is-ready' : ''}`}><i />{modelLabel}</span><span className="local-chip">LOCAL</span></div></header>
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>
}

function Panel({ title, subtitle, action, children, className = '' }) {
  return <section className={`panel ${className}`}><div className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>
}

function Stat({ label, value, suffix = '', note, tone = 'blue' }) {
  const display = value == null ? 'Not available' : typeof value === 'string' ? value : `${fmt(value)}${suffix}`
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone}`}><Activity size={17} /></span></div><div className="stat-value">{display}</div><div className="stat-foot">{note || 'From ingested historical records'}</div></article>
}

function Table({ columns, rows, empty = 'No records available.' }) {
  return <div className="table-scroll"><table><thead><tr>{columns.map(col => <th key={col.label}>{col.label}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, i) => <tr key={row.disruption_id || row.id || i}>{columns.map(col => <td key={col.key}>{col.render ? col.render(row[col.key], row) : row[col.key] ?? '—'}</td>)}</tr>) : <tr><td className="empty-table" colSpan={columns.length}>{empty}</td></tr>}</tbody></table></div>
}

function Overview({ onAnalyze }) {
  const [overview, setOverview] = useState(null); const [types, setTypes] = useState(null); const [disruptions, setDisruptions] = useState(null); const [activity, setActivity] = useState(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(true)
  const load = async () => { setLoading(true); setError(''); try { const [a,b,c] = await Promise.all([api.get('/analytics/overview'), api.get('/analytics/disruptions'), api.get('/disruptions?limit=8')]); setOverview(a.data); setTypes(b.data); setDisruptions(c.data) } catch (err) { setError(friendlyError(err)) } finally { setLoading(false) } }
  useEffect(() => { load() }, [])
  useEffect(() => { api.get('/history/stats').then(({data}) => setActivity(data)).catch(() => setActivity(null)) }, [])
  const categoryData = types?.by_type?.map(x => ({ name: x.label, count: x.count })) || []
  const severityData = types?.by_severity?.map(x => ({ name: `Severity ${x.label}`, count: x.count })) || []
  return <><PageHeading eyebrow={today} title="Supply chain overview" description="Disruption exposure, operational outcomes and supplier resilience." action={<button className="primary-btn" onClick={onAnalyze}><Sparkles size={16} /> Analyze disruption</button>} />
    {activity && <div className="stats-grid overview-account-stats"><Stat label="My analyses" value={activity.total_analyses} note="Your analysis history"/><Stat label="Claude" value={activity.provider_counts?.['Anthropic Claude'] || activity.provider_counts?.Anthropic || 0} note="Primary provider"/><Stat label="Gemini" value={activity.provider_counts?.['Google Gemini'] || activity.provider_counts?.Gemini || 0} note="First fallback"/><Stat label="Deterministic" value={activity.provider_counts?.['Deterministic Fallback'] || 0} note="Final fallback"/><Stat label="Used fallback" value={activity.fallback_requests} note="Your analyses"/></div>}
    {error && <ErrorState title="Overview data is unavailable" description={error} onRetry={load}/>}
    {loading ? <div className="overview-loading"><LoadingSkeleton rows={4} label="Loading supply chain overview"/><LoadingSkeleton rows={2} label="Loading historical trends"/></div> : !error && <>
      {overview && <div className="stats-grid"><Stat label="Total disruptions" value={overview.total_disruptions} /><Stat label="Average production impact" value={overview.average_production_impact_pct} suffix="%" tone="amber"/><Stat label="Total revenue loss" value={money(overview.total_revenue_loss_usd)} note="Sum across ingested historical records" tone="red"/><Stat label="Average response time" value={overview.average_response_time_days} suffix=" days"/><Stat label="Average recovery time" value={overview.average_recovery_time_days} suffix=" days" tone="green"/><Stat label="Backup supplier coverage" value={overview.backup_supplier_coverage_pct} suffix="%"/><Stat label="Permanent supplier change" value={overview.permanent_supplier_change_rate_pct} suffix="%" tone="amber"/></div>}
      <div className="overview-grid"><Panel title="Disruptions by type" subtitle="Actual records grouped by event type"><div className="dashboard-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={categoryData} margin={{ top: 10, right: 10, left: -17, bottom: 28 }}><CartesianGrid stroke="var(--line-soft)" vertical={false} /><XAxis dataKey="name" angle={-22} textAnchor="end" interval={0} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} /><YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)' }} /><Tooltip /><Bar dataKey="count" fill="var(--accent)" radius={[5,5,0,0]} /></BarChart></ResponsiveContainer></div></Panel><Panel title="Severity distribution" subtitle="Counts by recorded severity"><div className="dashboard-chart"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={severityData} dataKey="count" nameKey="name" innerRadius={52} outerRadius={86} paddingAngle={3}>{severityData.map((_, i) => <Cell key={i} fill={['#8b9af1','#7486eb','#5f72dc','#d39d54','#d76f68'][i % 5]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></div></Panel></div>
      <Panel title="Recent historical records" subtitle="Source dataset records stored in MySQL" className="disruption-panel"><Table columns={[{label:'ID',key:'disruption_id'},{label:'TYPE',key:'disruption_type'},{label:'INDUSTRY',key:'industry'},{label:'REGION',key:'supplier_region'},{label:'SEVERITY',key:'disruption_severity'},{label:'IMPACT',key:'production_impact_pct',render:v=>`${fmt(v)}%`},{label:'RESPONSE',key:'response_type'}]} rows={disruptions?.results || []} empty="No dataset records ingested yet." /></Panel>
    </>}
  </>
}

const initialForm = { description: 'A severe port disruption has delayed an electronics supplier in Asia-Pacific.', disruption_type: 'Port Congestion', industry: 'Electronics', supplier_tier: '2', supplier_region: 'Asia-Pacific', supplier_size: 'Medium', has_backup_supplier: false, disruption_severity: 4, production_impact_pct: '', revenue_loss_usd: '', response_type: '', response_time_days: '' }
function AnalysisForm({ onComplete, busy, setBusy }) {
  const [form, setForm] = useState(initialForm); const [error, setError] = useState('')
  const set = (key, value) => setForm(current => ({ ...current, [key]: value }))
  const submit = async event => {
    event.preventDefault(); setBusy(true); setError('')
    const payload = { ...form, supplier_tier: form.supplier_tier, production_impact_pct: form.production_impact_pct === '' ? null : Number(form.production_impact_pct), revenue_loss_usd: form.revenue_loss_usd === '' ? null : Number(form.revenue_loss_usd), response_type: form.response_type || null, response_time_days: form.response_time_days === '' ? null : Number(form.response_time_days) }
    try { const { data } = await api.post('/disruptions/analyze', payload); onComplete(data) } catch (err) { setError(friendlyError(err)) } finally { setBusy(false) }
  }
  return <form className={`panel analysis-form ${busy ? 'is-processing' : ''}`} onSubmit={submit} aria-busy={busy}>
    <div className="panel-header"><div><span className="form-step">INCIDENT INTAKE</span><h2>Current disruption</h2><p>Describe the observed event. Fields are validated by the API.</p></div></div>
    <label>Event details<textarea required minLength="12" maxLength="1200" value={form.description} onChange={e => set('description',e.target.value)} placeholder="Describe the observed disruption and its operational context." /></label>
    <div className="form-row"><label>Disruption type<select value={form.disruption_type} onChange={e => set('disruption_type',e.target.value)}>{disruptionTypes.map(x=><option key={x}>{x}</option>)}</select></label><label>Industry<select value={form.industry} onChange={e => set('industry',e.target.value)}>{industries.map(x=><option key={x}>{x}</option>)}</select></label></div>
    <div className="form-row"><label>Supplier tier<select value={form.supplier_tier} onChange={e => set('supplier_tier',e.target.value)}>{['1','2','3','4'].map(x=><option key={x}>{x}</option>)}</select></label><label>Supplier region<select value={form.supplier_region} onChange={e => set('supplier_region',e.target.value)}>{regions.map(x=><option key={x}>{x}</option>)}</select></label></div>
    <div className="form-row"><label>Supplier size<select value={form.supplier_size} onChange={e => set('supplier_size',e.target.value)}>{sizes.map(x=><option key={x}>{x}</option>)}</select></label><label>Backup supplier<select value={String(form.has_backup_supplier)} onChange={e => set('has_backup_supplier',e.target.value === 'true')}><option value="true">Yes</option><option value="false">No</option></select></label></div>
    <div className="form-row"><label>Severity <b>{form.disruption_severity} / 5</b><input type="range" min="1" max="5" value={form.disruption_severity} onChange={e => set('disruption_severity',Number(e.target.value))} /></label><label>Observed production impact (%)<input type="number" min="0" max="100" step="0.1" value={form.production_impact_pct} onChange={e => set('production_impact_pct',e.target.value)} placeholder="Optional if not yet known" /></label></div>
    <div className="form-row"><label>Observed revenue loss (USD)<input type="number" min="0" step="0.01" value={form.revenue_loss_usd} onChange={e => set('revenue_loss_usd',e.target.value)} placeholder="Optional if not yet known" /></label><label>Response strategy, if known<select value={form.response_type} onChange={e => set('response_type',e.target.value)}><option value="">Not known yet</option>{responseTypes.map(x=><option key={x}>{x}</option>)}</select></label></div>
    <div className="form-row"><label>Response time (days), if known<input type="number" min="0" max="5000" value={form.response_time_days} onChange={e => set('response_time_days',e.target.value)} /></label><div className="input-note">Later response and recovery fields are excluded from initial prediction features.</div></div>
    {error && <ErrorState title="Analysis could not be completed" description={error}/ >}
    <button className="primary-btn run-btn" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={16}/> Analysis running</> : 'Analyze disruption'} {!busy && <ArrowUpRight size={15}/>}</button>
  </form>
}

function PriorityBadge({ level }) { const normalized = (level || 'unavailable').toLowerCase(); return <span className={`risk-badge ${normalized}`}><i/>{level || 'Priority unavailable'}</span> }
function MetricTile({ label, value, tone = '', uncertainty = null }) {
  return (
    <article className={`metric-tile ${tone}`}>
      <span>{label}</span>
      <b>{value}</b>
      {uncertainty?.display_interval && (
        <small style={{ fontSize: '9.5px', color: 'var(--text-muted)', marginTop: '3px', display: 'block' }}>
          Interval: {uncertainty.display_interval}
        </small>
      )}
    </article>
  )
}

function AnalysisSummary({ result, busy }) {
  if (busy) return <AnalysisProgress />
  if (!result) return <section className="panel result-empty"><div className="empty-orbit"><Sparkles size={22} /></div><h2>Ready to analyze disruption</h2><p>Submit an event to retrieve historical cases and compare predictions.</p><div className="workflow-tags"><span>Classification</span><span>Historical evidence</span><span>Predictions</span></div></section>
  const preds = result.predictions?.predictions || {}
  const supplierChange = preds.permanent_supplier_change
  return <div className="analysis-summary">
    <section className="priority-result">
      <div><span className="result-kicker"><i /> ANALYSIS COMPLETE</span><h2>{result.classification?.disruption_type} <span>·</span> {result.classification?.industry}</h2><p>Severity {result.classification?.severity}/5</p></div>
      <PriorityBadge level={result.priority?.priority_level} />
      <div className="priority-score-bar"><span>Priority score</span><b>{result.priority?.priority_score == null ? 'Not available' : `${result.priority.priority_score}/100`}</b><i><em style={{ width: `${Math.max(0, Math.min(100, result.priority?.priority_score || 0))}%` }} /></i></div>
    </section>
    <div className="result-grid">
      <MetricTile label="Production impact" value={preds.production_impact?.prediction == null ? 'Not available' : `${fmt(preds.production_impact.prediction)}%`} tone="impact" uncertainty={preds.production_impact?.uncertainty} />
      <MetricTile label="Financial loss" value={money(preds.financial_loss?.prediction)} tone="loss" uncertainty={preds.financial_loss?.uncertainty} />
      <MetricTile label="Predicted recovery" value={preds.recovery_time?.prediction == null ? 'Not available' : `${fmt(preds.recovery_time.prediction)} days`} tone="recovery" uncertainty={preds.recovery_time?.uncertainty} />
      <MetricTile label="Permanent supplier change" value={supplierChange?.available === false ? 'Model not trained' : supplierChange?.prediction == null ? 'Not available' : `${supplierChange.prediction ? 'Yes' : 'No'} · score ${fmt((supplierChange.probability || 0) * 100)}%`} />
    </div>
  </div>
}

function ExplanationCard({ target, features = [] }) {
  const [showAll, setShowAll] = useState(false)
  const positive = features.filter(item => Number(item.importance) > 0)
  const visible = showAll ? positive : positive.slice(0, 4)
  const maximum = Math.max(...positive.map(item => Number(item.importance)), 0)
  return <article className="explanation-card"><h3>{target.replaceAll('_', ' ')}</h3>{visible.length ? <div className="explanation-factors">{visible.map(item => <div className="explanation-factor" key={item.feature}><div><span>{item.feature}</span><b>{fmt(item.importance)}</b></div><i><em style={{ width: `${maximum ? Number(item.importance) / maximum * 100 : 0}%` }} /></i></div>)}</div> : <p className="explanation-empty">No positive feature importance was returned.</p>}{positive.length > 4 && <button type="button" className="text-link" aria-expanded={showAll} onClick={() => setShowAll(value => !value)}>{showAll ? 'Show fewer' : `Show ${positive.length - 4} more`} <ChevronRight size={13} /></button>}</article>
}

function AnalysisRawResponse({ result }) {
  const [formatted, setFormatted] = useState(true)
  const [copyStatus, setCopyStatus] = useState('')
  const copyJson = async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(result, null, 2)); setCopyStatus('JSON copied') }
    catch { setCopyStatus('Copy unavailable in this browser') }
  }
  return <details className="panel analysis-raw"><summary><span>Full API response</span><small>Developer and debug information</small></summary><div className="analysis-raw-controls"><button type="button" className="outline-btn" onClick={copyJson}><Copy size={13} />{copyStatus || 'Copy JSON'}</button><label><input type="checkbox" checked={formatted} onChange={event => setFormatted(event.target.checked)} />View formatted JSON</label></div><pre className="json-panel">{JSON.stringify(result, null, formatted ? 2 : 0)}</pre></details>
}

function EvaluationSection({ result }) {
  const evalData = result?.evaluation
  if (!evalData) return null

  const ragas = evalData.ragas || {}
  const overallScore = ragas.overall_pct ?? evalData.overall_score_pct
  const status = evalData.status || (overallScore != null ? 'GOOD' : 'UNAVAILABLE')
  const statusClass = status.toLowerCase().replace(/\s+/g, '-')
  const hasWarning = evalData.has_warning || false
  const warnings = evalData.warnings || []
  const metrics = ragas.metrics || {}
  const hasMetrics = Object.keys(metrics).length > 0
  const isUnavailable = ragas.status === 'unavailable' || overallScore == null

  return (
    <section className="eval-panel" id="evaluation-strategy-section">
      <div className="eval-header">
        <div>
          <span className="eyebrow">RAGAS EVALUATION</span>
          <h2><Database size={19} color="var(--accent)" /> RAGAS Quality Assessment</h2>
          <p>Automated evaluation of recommendation faithfulness, answer relevancy, and retrieval context quality via the RAGAS framework.</p>
        </div>
      </div>

      <div className="eval-overall-card">
        <div className="eval-score-circle" style={{ borderColor: evalData.status_color || (isUnavailable ? 'var(--danger)' : 'var(--accent)') }}>
          <span className="eval-score-num">{isUnavailable ? 'N/A' : `${overallScore}%`}</span>
          <span className="eval-score-lbl">RAGAS</span>
        </div>
        <div className="eval-overall-info">
          <div className="eval-status-row">
            <span className={`eval-status-badge ${statusClass}`}>{status}</span>
            <span className="eval-threshold-text">Target Threshold: <b>{evalData.threshold_pct ?? 75}%</b></span>
            <span className="local-chip">Framework: RAGAS</span>
          </div>
          <div className="eval-weights-list">
            <span className="eval-weight-pill">Faithfulness</span>
            <span className="eval-weight-pill">Answer Relevancy</span>
            <span className="eval-weight-pill">Context Precision</span>
            <span className="eval-weight-pill">Context Recall</span>
          </div>
        </div>
      </div>

      {hasWarning && warnings.length > 0 && (
        <div className="eval-warning-banner" role="alert">
          <AlertTriangle size={20} />
          <div>
            <b>RAGAS Evaluation Warning</b>
            <ul>
              {warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </div>
        </div>
      )}

      {isUnavailable || !hasMetrics ? (
        <div className="empty-state panel" style={{ padding: '24px' }}>
          <Database size={24} color="var(--warning)" />
          <b>RAGAS Evaluation: Unavailable</b>
          <span>{ragas.error || 'RAGAS evaluation could not be completed for this request. The main disruption analysis is fully delivered and usable.'}</span>
        </div>
      ) : (
        <div className="eval-cards-grid">
          {Object.entries(metrics).map(([key, item]) => (
            <div className="eval-card" key={key}>
              <div className="eval-card-top">
                <span className="eval-card-title">{key.replace(/_/g, ' ')}</span>
                <span className={item.passed ? 'eval-badge-pass' : 'eval-badge-fail'}>{item.status}</span>
              </div>
              <span className="eval-card-score">{item.score_pct}%</span>
              <div className="eval-progress-bar">
                <div
                  className="eval-progress-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, item.score_pct || 0))}%`,
                    backgroundColor: item.passed ? 'var(--success)' : 'var(--danger)',
                  }}
                />
              </div>
              <p className="eval-card-reason">{item.reason}</p>
              <div className="eval-card-footer">
                <span>Target: {item.threshold_pct}%</span>
                <span>Score: {item.score}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

function PredictionReliabilityCard({ reliability, predictions }) {
  if (!reliability && !predictions) return null
  const rel = reliability || {}
  const score = rel.reliability_score || 80
  const level = rel.reliability_level || (score >= 80 ? 'HIGH' : score >= 65 ? 'MODERATE' : 'LOW')
  const preds = predictions || {}

  return (
    <Panel
      title="Prediction Reliability & Uncertainty Intervals"
      subtitle="Derived from held-out validation test split (15,000 cases); distinct from LLM confidence"
      className="reliability-card"
    >
      <div className="reliability-header">
        <div>
          <span className={`reliability-badge ${level.toLowerCase()}`}>
            {level === 'HIGH' ? <Check size={13} /> : <AlertTriangle size={13} />}
            {level} RELIABILITY ({score}%)
          </span>
          <span className="uncertainty-sub" style={{ marginTop: '4px' }}>
            {rel.human_review_recommended
              ? 'Human Review Recommended: Operational uncertainty exceeds standard thresholds.'
              : 'Model predictions within standard test operating bounds.'}
          </span>
        </div>
        <div style={{ textAlign: 'right', fontSize: '9px', color: 'var(--text-muted)' }}>
          <div>Evaluation Split: 15,000 held-out cases</div>
          <div>Models: Pre-trained Random Forest</div>
        </div>
      </div>

      <div className="stats-grid" style={{ marginTop: '10px' }}>
        <article className="stat-card">
          <div className="stat-top"><span>Production Impact</span></div>
          <div className="stat-value">{preds.production_impact?.prediction != null ? `${preds.production_impact.prediction}%` : '—'}</div>
          <div className="metric-uncertainty-pill">
            {preds.production_impact?.uncertainty?.display_interval
              ? `Range: ${preds.production_impact.uncertainty.display_interval}`
              : '±8.0% MAE (R² 0.68)'}
          </div>
          <div className="stat-foot">Test MAE: 7.99% · Test RMSE: 9.92%</div>
        </article>

        <article className="stat-card">
          <div className="stat-top"><span>Financial Loss</span></div>
          <div className="stat-value">{preds.financial_loss?.prediction != null ? money(preds.financial_loss.prediction) : '—'}</div>
          <div className="metric-uncertainty-pill">
            {preds.financial_loss?.uncertainty?.test_mae
              ? `±${money(preds.financial_loss.uncertainty.test_mae)} MAE`
              : '±$1.77M MAE (R² 0.37)'}
          </div>
          <div className="stat-foot">Test MAE: $1.77M · Test RMSE: $4.42M</div>
        </article>

        <article className="stat-card">
          <div className="stat-top"><span>Recovery Time</span></div>
          <div className="stat-value">{preds.recovery_time?.prediction != null ? `${preds.recovery_time.prediction} days` : '—'}</div>
          <div className="metric-uncertainty-pill">
            {preds.recovery_time?.uncertainty?.display_interval
              ? `Range: ${preds.recovery_time.uncertainty.display_interval}`
              : '±31.7d MAE (R² 0.37)'}
          </div>
          <div className="stat-foot">Test MAE: 31.7d · Test RMSE: 45.2d</div>
        </article>

        <article className="stat-card">
          <div className="stat-top"><span>Supplier Replacement</span></div>
          <div className="stat-value">{preds.permanent_supplier_change?.prediction ? 'Yes' : 'No'}</div>
          <div className="metric-uncertainty-pill">
            {preds.permanent_supplier_change?.probability != null
              ? `Score ${Math.round(preds.permanent_supplier_change.probability * 100)}%`
              : 'Test Acc 64.0%'}
          </div>
          <div className="stat-foot">ROC-AUC: 0.635 · Uncalibrated score</div>
        </article>
      </div>

      {rel.reasons?.length > 0 && (
        <div style={{ marginTop: '10px' }}>
          <b style={{ fontSize: '10px', color: 'var(--ink)' }}>Reliability Assessment Factors:</b>
          <ul className="reliability-reasons-list" style={{ marginTop: '6px' }}>
            {rel.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '8px', borderTop: '1px solid var(--line-soft)', paddingTop: '8px' }}>
        <b>Notice:</b> Model performance metrics (R², MAE, RMSE) and empirical uncertainty bounds reflect held-out test data. Retrieved case volume and LLM self-reported confidence do not constitute calibrated prediction probabilities.
      </div>
    </Panel>
  )
}

function WhatIfSimulator({ simulationData, onSelectRecord }) {
  const [activeStrategy, setActiveStrategy] = useState(null)
  const list = simulationData || []
  if (!list.length) return null

  const selected = activeStrategy ? list.find(s => s.response_type === activeStrategy) || list[0] : list[0]

  return (
    <Panel
      title="What-if Disruption Simulator"
      subtitle="Compare candidate response strategies using historical peer incident outcomes"
      className="whatif-section"
    >
      <div className="whatif-notice">
        <CircleHelp size={16} />
        <div>
          <b>Historical Empirical Statistics:</b> Compares observed recovery outcomes across retrieved past incidents. Machine learning predictions represent pre-response impact estimations. Changing response strategies does not guarantee an identical future outcome.
        </div>
      </div>

      <div className="whatif-strategy-grid">
        {list.map(strat => {
          const isRec = strat.is_recommended
          const isAct = strat.response_type === selected?.response_type
          return (
            <div
              key={strat.response_type}
              className={`whatif-card ${isRec ? 'is-recommended' : ''} ${isAct ? 'is-active' : ''}`}
              onClick={() => setActiveStrategy(strat.response_type)}
            >
              <div className="whatif-header">
                <span className="whatif-title">{strat.response_type}</span>
                <span className="whatif-score-badge">
                  Score {strat.effectiveness_score}/100
                </span>
              </div>

              <div className="whatif-metrics">
                <div className="whatif-metric-item">
                  <small>Median Recovery</small>
                  <b>{strat.historical_median_full_recovery_days != null ? `${strat.historical_median_full_recovery_days} d` : '—'}</b>
                </div>
                <div className="whatif-metric-item">
                  <small>Median Impact</small>
                  <b>{strat.historical_median_production_impact_pct != null ? `${strat.historical_median_production_impact_pct}%` : '—'}</b>
                </div>
                <div className="whatif-metric-item">
                  <small>Median Loss</small>
                  <b>{strat.historical_median_revenue_loss_usd != null ? money(strat.historical_median_revenue_loss_usd) : '—'}</b>
                </div>
              </div>

              <div className="whatif-tradeoff-summary">
                {strat.trade_offs}
              </div>

              <div className="whatif-detail-tray">
                <span>{strat.data_source}</span>
                {strat.supporting_disruption_ids?.length > 0 && (
                  <div className="whatif-ids-row">
                    <small>Evidence IDs:</small>
                    {strat.supporting_disruption_ids.slice(0, 4).map(id => (
                      <span
                        key={id}
                        className="whatif-id-pill"
                        style={{ cursor: 'pointer' }}
                        onClick={e => { e.stopPropagation(); onSelectRecord && onSelectRecord({ disruption_id: id }) }}
                      >
                        {id}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {selected && (
        <div className="whatif-procon-box" style={{ marginTop: '16px' }}>
          <div className="whatif-procon-col">
            <b style={{ color: 'var(--success)' }}>Operational Advantages</b>
            <ul>
              {selected.advantages?.map((adv, i) => <li key={i}>{adv}</li>)}
            </ul>
          </div>
          <div className="whatif-procon-col">
            <b style={{ color: 'var(--danger)' }}>Disadvantages & Friction</b>
            <ul>
              {selected.disadvantages?.map((dis, i) => <li key={i}>{dis}</li>)}
            </ul>
          </div>
        </div>
      )}
    </Panel>
  )
}

function AnalysisDetails({ result }) {
  const [selectedRecord, setSelectedRecord] = useState(null)
  const rec = result.recommendation || {}
  const preds = result.predictions?.predictions || {}
  const tracking = result.provider_tracking || {}
  const cases = result.similar_cases || []
  const featureGroups = Object.entries(result.explanation?.impact_feature_importance || {})
  return <div className="analysis-details">
    <Panel title="Generation details" subtitle="Provider and model used for this analysis"><div className="generation-grid"><span><small>Provider</small><b>{tracking.final_provider || 'Provider unavailable'}</b></span><span><small>Model</small><b>{tracking.final_model || 'Model unavailable'}</b></span><span><small>Fallback used</small><b>{tracking.fallback_used ? 'Yes' : 'No'}</b></span><span><small>Workflow status</small><b>{result.llm_status || 'Unavailable'}</b></span></div></Panel>
    <section className="recommendation recommendation-featured analysis-recommendation">
      <span className="recommend-icon"><Sparkles size={17} /></span>
      <div className="recommendation-content">
        <span className="recommend-kicker">RECOMMENDED RESPONSE</span>
        <b>{rec.recommended_response || 'Insufficient historical evidence'}</b>
        <p className="recommendation-reason">{rec.summary || rec.reasoning?.[0] || 'Run an analysis with available historical cases to compare response strategies.'}</p>
        {rec.reasoning?.length > 0 && <details className="recommendation-reasoning"><summary>Why this response?</summary><p>{rec.reasoning.join(' ')}</p></details>}
        <div className="recommendation-facts">
          <span><small>Historical evidence</small><b>{fmt(rec.confidence == null ? null : rec.confidence * 100)}%</b></span>
          <span><small>Predicted recovery</small><b>{preds.recovery_time?.prediction == null ? 'Not available' : `${fmt(preds.recovery_time.prediction)} days`}</b></span>
          <span><small>Financial impact</small><b>{money(preds.financial_loss?.prediction)}</b></span>
          <span className={rec.human_review_required ? 'review-required' : 'review-ok'}><small>Human review</small><b>{rec.human_review_required ? 'Required' : 'Not required'}</b></span>
        </div>
        {rec.supporting_disruption_ids?.length > 0 && (
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <small style={{ color: 'var(--text-muted)', fontSize: '10.5px', fontWeight: 600 }}>Supporting Evidence Cases:</small>
            {rec.supporting_disruption_ids.map(id => (
              <span
                key={id}
                className="whatif-id-pill"
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedRecord({ disruption_id: id })}
                title="Click to inspect historical peer case"
              >
                {id}
              </span>
            ))}
          </div>
        )}
      </div>
    </section>

    {result.what_if_simulation?.length > 0 && (
      <WhatIfSimulator simulationData={result.what_if_simulation} onSelectRecord={setSelectedRecord} />
    )}

    <PredictionReliabilityCard reliability={result.predictions?.reliability} predictions={result.predictions?.predictions} />

    <Panel title="Historical evidence" subtitle={`${cases.length} retrieved cases · select a case to inspect its outcomes`} className="analysis-evidence-panel">{cases.length ? <div className="analysis-evidence-grid">{cases.map(item => <HistoricalCaseCard item={item} compact key={item.disruption_id} onSelect={() => setSelectedRecord(item)} />)}</div> : <div className="empty-state"><Network size={18} /><b>No historical cases returned</b><span>Recommendations should be treated as unavailable without retrieved evidence.</span></div>}</Panel>
    <Panel title="Response comparison" subtitle="Outcomes calculated from the retrieved records" className="analysis-comparison-panel"><Table columns={[{label:'RESPONSE',key:'response_type'},{label:'CASES',key:'case_count'},{label:'MEDIAN IMPACT',key:'median_production_impact_pct',render:v=>`${fmt(v)}%`},{label:'MEDIAN LOSS',key:'median_revenue_loss_usd',render:money},{label:'MEDIAN RECOVERY',key:'median_full_recovery_days',render:v=>`${fmt(v)} days`}]} rows={result.evidence || []} empty="No response comparison available." /></Panel>
    <Panel title="Why these predictions?" subtitle="Global feature importance from the fitted models; not a causal explanation"><div className="explanation-grid">{featureGroups.map(([target, features]) => <ExplanationCard key={target} target={target} features={features} />)}</div>{!featureGroups.length && <div className="empty-state">Feature importance is not available for this response.</div>}</Panel>
    {result.priority?.available && (
      <Panel title="Disruption priority assessment" subtitle={`Priority score: ${result.priority.priority_score}/100 · ${result.priority.priority_level} priority`}>
        <div className="generation-grid">
          <span><small>Priority score</small><b>{result.priority.priority_score}/100</b></span>
          <span><small>Priority level</small><b><PriorityBadge level={result.priority.priority_level} /></b></span>
          <span><small>Disruption severity</small><b>{result.classification?.severity}/5</b></span>
          <span><small>Calculation method</small><b>Empirical percentile weighted model</b></span>
        </div>
        {result.priority.priority_reasons?.length > 0 && (
          <div className="factor-list" style={{ marginTop: '12px' }}>
            <div className="factor-head"><b>Assessment factor</b><span>Observed value & dataset percentile</span></div>
            {result.priority.priority_reasons.map((reason, i) => (
              <div className="factor-row" key={i}>
                <span>{reason.factor} (weight {Math.round(reason.weight * 100)}%)</span>
                <b>{reason.factor.toLowerCase().includes('loss') ? money(reason.observed) : reason.factor.toLowerCase().includes('impact') ? `${reason.observed}%` : `${reason.observed}/5`} · {reason.dataset_percentile}th percentile</b>
              </div>
            ))}
          </div>
        )}
      </Panel>
    )}
    <Panel title="Recovery and resilience" subtitle="Prediction and retrieved-case context"><div className="generation-grid recovery-grid"><span><small>Predicted recovery</small><b>{result.recovery_analysis?.predicted_full_recovery_days == null ? 'Not available' : `${fmt(result.recovery_analysis.predicted_full_recovery_days)} days`}</b></span><span><small>Historical median</small><b>{result.recovery_analysis?.retrieved_case_median_days == null ? 'Not available' : `${fmt(result.recovery_analysis.retrieved_case_median_days)} days`}</b></span><span><small>Resilience score</small><b>{result.resilience_analysis?.resilience_score == null ? result.resilience_analysis?.message || 'Not available' : `${fmt(result.resilience_analysis.resilience_score)} / 100`}</b></span><span><small>Risk level</small><b>{result.resilience_analysis?.risk_level || 'Not available'}</b></span></div></Panel>
    <EvaluationSection result={result} />
    <AnalysisRawResponse result={result} />
    <div className="analysis-validation"><span className={result.validation?.valid ? 'validation-ok' : 'validation-review'}>{result.validation?.valid ? <Check size={15} /> : <AlertTriangle size={15} />}{result.validation?.valid ? ' Evidence checks passed' : ' Review required'}</span><span>{result.llm_status || 'LLM not configured'}{result.validation?.errors?.length ? ` · ${result.validation.errors.join(' ')}` : ''}</span></div>
    {selectedRecord && <HistoricalRecordModal item={selectedRecord} onClose={() => setSelectedRecord(null)} />}
  </div>
}

function AnalysisPage({ latest, onComplete, onToast }) {
  const [busy, setBusy] = useState(false)
  return <><PageHeading eyebrow="DECISION WORKFLOW" title="Disruption analysis" description="Assess an event with historical evidence and saved prediction models."/><div className="analysis-grid analysis-hero-grid"><AnalysisForm busy={busy} setBusy={setBusy} onComplete={result=>{onComplete(result);onToast('Analysis completed')}}/><AnalysisSummary result={latest} busy={busy}/></div>{latest && <AnalysisDetails result={latest}/>}</>
}

function SimilarCases() {
  const [query,setQuery]=useState('Severe supplier disruption in Asia-Pacific affecting electronics production'); const [topK,setTopK]=useState(10); const [filters,setFilters]=useState({industry:'',supplier_tier:'',supplier_region:'',disruption_type:'',supplier_size:'',has_backup_supplier:'',disruption_severity:'',response_type:''}); const [rows,setRows]=useState([]); const [selectedRecord,setSelectedRecord]=useState(null); const [error,setError]=useState(''); const [busy,setBusy]=useState(false); const [searched,setSearched]=useState(false)
  const search=async e=>{e.preventDefault();setBusy(true);setError('');setSearched(true);const selected=Object.fromEntries(Object.entries(filters).filter(([,v])=>v!==''&&v!==null));if(selected.has_backup_supplier!==undefined)selected.has_backup_supplier=selected.has_backup_supplier==='true';if(selected.disruption_severity!==undefined)selected.disruption_severity={$gte:Number(selected.disruption_severity)};try{const {data}=await api.post('/retrieval/similar',{query,filters:selected,top_k:topK});setRows(data.results)}catch(err){setError(friendlyError(err));setRows([])}finally{setBusy(false)}}
  const select=(key,label,choices)=><label>{label}<select value={filters[key]} onChange={e=>setFilters({...filters,[key]:e.target.value})}><option value="">Any</option>{choices.map(x=><option value={x} key={x}>{x}</option>)}</select></label>
  return <><PageHeading eyebrow="HISTORICAL EVIDENCE" title="Similar disruptions" description="Search the indexed records and compare real response and recovery outcomes."/><form className="panel filter-form" onSubmit={search}><label className="search-query-label">Search description<input value={query} onChange={e=>setQuery(e.target.value)} minLength="3" aria-label="Historical disruption search query"/></label><label>Number of cases<b>{topK}</b><input type="range" min="1" max="20" value={topK} onChange={e=>setTopK(Number(e.target.value))}/></label>{select('disruption_type','Disruption type',disruptionTypes)}{select('industry','Industry',industries)}{select('supplier_tier','Supplier tier',['1','2','3','4'])}{select('supplier_region','Supplier region',regions)}{select('supplier_size','Supplier size',sizes)}<label>Backup supplier<select value={filters.has_backup_supplier} onChange={e=>setFilters({...filters,has_backup_supplier:e.target.value})}><option value="">Any</option><option value="true">Yes</option><option value="false">No</option></select></label><label>Minimum severity<select value={filters.disruption_severity} onChange={e=>setFilters({...filters,disruption_severity:e.target.value})}><option value="">Any</option>{[1,2,3,4,5].map(x=><option value={x} key={x}>{x}+</option>)}</select></label>{select('response_type','Historical response',responseTypes)}<button className="primary-btn" disabled={busy}><Search size={15}/>{busy?'Searching records…':'Search cases'}</button></form>
    {error&&<ErrorState title="Historical search failed" description={error}/>}
    <Panel title={busy?'Searching historical disruptions…':`Retrieved cases${searched?` · ${rows.length}`:''}`} subtitle="Select a record for impact, response and recovery details">{busy?<LoadingSkeleton rows={4} label="Searching historical disruptions"/>:rows.length?<div className="historical-list">{rows.map(item=><HistoricalCaseCard item={item} key={item.disruption_id} onSelect={()=>setSelectedRecord(item)}/>)}</div>:searched&&!error?<div className="empty-state"><Search size={20}/><b>No similar historical disruptions found</b><span>Try a broader description, remove filters, or change the disruption details.</span></div>:<div className="empty-state"><Network size={20}/><b>Search historical disruptions</b><span>Results will include only records returned by the retrieval service.</span></div>}</Panel>
    {selectedRecord && <HistoricalRecordModal item={selectedRecord} onClose={()=>setSelectedRecord(null)}/>}
  </>
}

function PredictionPage() {
  const [values,setValues]=useState({disruption_type:'Port Congestion',industry:'Electronics',supplier_tier:'2',supplier_region:'Asia-Pacific',supplier_size:'Medium',has_backup_supplier:true,disruption_severity:4}); const [result,setResult]=useState(null); const [error,setError]=useState(''); const [busy,setBusy]=useState(false)
  const predict=async e=>{e.preventDefault();setError('');setResult(null);setBusy(true);try{const {data}=await api.post('/predictions',values);setResult(data)}catch(err){setError(friendlyError(err))}finally{setBusy(false)}}
  const set=(key,value)=>setValues({...values,[key]:value})
  return <><PageHeading eyebrow="SAVED PREDICTION MODELS" title="Predictions" description="Inference uses persisted models with empirical uncertainty bounds."/><form className="panel prediction-form" onSubmit={predict}><div className="form-row"><label>Disruption type<select value={values.disruption_type} onChange={e=>set('disruption_type',e.target.value)}>{disruptionTypes.map(x=><option key={x}>{x}</option>)}</select></label><label>Industry<select value={values.industry} onChange={e=>set('industry',e.target.value)}>{industries.map(x=><option key={x}>{x}</option>)}</select></label><label>Supplier tier<select value={values.supplier_tier} onChange={e=>set('supplier_tier',e.target.value)}>{['1','2','3','4'].map(x=><option key={x}>{x}</option>)}</select></label><label>Region<select value={values.supplier_region} onChange={e=>set('supplier_region',e.target.value)}>{regions.map(x=><option key={x}>{x}</option>)}</select></label><label>Supplier size<select value={values.supplier_size} onChange={e=>set('supplier_size',e.target.value)}>{sizes.map(x=><option key={x}>{x}</option>)}</select></label><label>Backup source<select value={String(values.has_backup_supplier)} onChange={e=>set('has_backup_supplier',e.target.value==='true')}><option value="true">Yes</option><option value="false">No</option></select></label><label>Severity<select value={values.disruption_severity} onChange={e=>set('disruption_severity',Number(e.target.value))}>{[1,2,3,4,5].map(x=><option key={x}>{x}</option>)}</select></label></div><button className="primary-btn" disabled={busy}>{busy?<><LoaderCircle className="spin" size={16}/> Running saved models</>:'Run saved models'}</button></form>
    {error&&<ErrorState title="Predictions are unavailable" description={error}/>}{busy&&<LoadingSkeleton rows={3} label="Loading saved model predictions"/>}
    {result&&<>
      <PredictionReliabilityCard reliability={result.reliability} predictions={result.predictions} />
      <Panel title="Model predictions" subtitle="Empirical uncertainty ranges reflect held-out test set performance. Global feature importance is not a causal explanation.">
        <div className="result-grid">
          {Object.entries(result.predictions).map(([key,value])=><MetricTile key={key} label={key.replaceAll('_',' ')} value={value.available===false?'Model not trained':key==='permanent_supplier_change'?`${value.prediction?'Yes':'No'} · score ${fmt((value.probability || 0)*100)}%`:value.unit==='USD'?money(value.prediction):`${fmt(value.prediction)} ${value.unit||''}`} uncertainty={value.uncertainty} />)}
        </div>
        <pre className="json-panel">{JSON.stringify(result.explanations,null,2)}</pre>
      </Panel>
    </>}
  </>
}

function Recommendations({latest}) {
  const rec=latest?.recommendation; const [evidenceOpen,setEvidenceOpen]=useState(false); const [selectedRecord,setSelectedRecord]=useState(null); const cases=latest?.similar_cases||[]
  return <><PageHeading eyebrow="EVIDENCE-GROUNDED RESPONSE" title="Recommendations" description="Strategies are selected from retrieved cases and their observed outcomes."/>{!rec?<div className="empty-state panel"><ClipboardList size={20}/><b>No saved recommendation yet</b><span>Run an analysis to generate a response based on retrieved historical evidence.</span></div>:<>
    <section className="recommendation recommendation-featured recommendation-page">
      <span className="recommend-icon"><Sparkles size={18}/></span>
      <div>
        <span className="recommend-kicker">RECOMMENDED RESPONSE</span>
        <b>{rec.recommended_response}</b>
        <p>{rec.reasoning?.join(' ')||'No additional reasoning returned.'}</p>
        <small>Evidence share: {fmt(rec.confidence==null?null:rec.confidence*100)}% · {rec.human_review_required?'Human review required':'Evidence threshold met'}</small>

        {rec.supporting_disruption_ids?.length > 0 && (
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <small style={{ color: 'var(--text-muted)', fontSize: '10.5px', fontWeight: 600 }}>Supporting Evidence IDs:</small>
            {rec.supporting_disruption_ids.map(id => (
              <span key={id} className="whatif-id-pill" style={{ cursor: 'pointer' }} onClick={() => setSelectedRecord({ disruption_id: id })} title="Click to view historical case details">
                {id}
              </span>
            ))}
          </div>
        )}
        <button className="text-link" style={{ marginTop: '8px' }} onClick={()=>setEvidenceOpen(v=>!v)} aria-expanded={evidenceOpen}>{evidenceOpen?'Hide':'View'} supporting cases <ChevronRight size={14}/></button>
      </div>
    </section>

    {rec.recovery_statistics && (
      <div className="stats-grid" style={{ marginTop: '14px' }}>
        <Stat label="Historical Median Recovery" value={rec.recovery_statistics.median_recovery_days != null ? `${rec.recovery_statistics.median_recovery_days} days` : '—'} note="Observed duration across supporting records" tone="green" />
        <Stat label="Historical Median Impact" value={rec.recovery_statistics.median_production_impact_pct != null ? `${rec.recovery_statistics.median_production_impact_pct}%` : '—'} note="Observed production drop" tone="amber" />
        <Stat label="Historical Median Loss" value={rec.recovery_statistics.median_revenue_loss_usd != null ? money(rec.recovery_statistics.median_revenue_loss_usd) : '—'} note="Observed financial loss" tone="red" />
        <Stat label="Supporting Evidence Cases" value={rec.recovery_statistics.case_count || '0'} note="Retrieved peer incident count" tone="blue" />
      </div>
    )}

    {latest?.what_if_simulation?.length > 0 && (
      <WhatIfSimulator simulationData={latest.what_if_simulation} onSelectRecord={setSelectedRecord} />
    )}

    {evidenceOpen&&<Panel title="Cases supporting this recommendation" subtitle="Select a record to inspect its outcomes"><div className="historical-list">{cases.length?cases.map(item=><HistoricalCaseCard item={item} key={item.disruption_id} onSelect={()=>setSelectedRecord(item)}/>):<div className="empty-state">No case-level evidence was returned.</div>}</div></Panel>}
    {selectedRecord&&<HistoricalRecordModal item={selectedRecord} onClose={()=>setSelectedRecord(null)}/>}

    <div className="stats-grid"><Stat label="Recommended response" value={rec.recommended_response} note="Selected from retrieved response strategies"/><Stat label="Evidence share" value={rec.confidence==null?null:rec.confidence*100} suffix="%" note={rec.confidence_basis}/><Stat label="Human review" value={rec.human_review_required?'Required':'Recommended'} note="Based on validation and evidence coverage" tone="amber"/></div>

    <Panel title="Reasoning and alternatives" subtitle="Historical case comparison">
      <ul className="plain-list">{(rec.reasoning||[]).map((line,i)=><li key={i}>{line}</li>)}</ul>

      {rec.alternative_trade_offs?.length > 0 ? (
        <div style={{ marginTop: '14px' }}>
          <b>Alternative Strategy Trade-Offs & Counter-Evidence</b>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '10px', marginTop: '10px' }}>
            {rec.alternative_trade_offs.map((alt, i) => (
              <div key={i} className="tradeoff-card" style={{ background: 'var(--panel-subtle)', padding: '12px', borderRadius: '8px', border: '1px solid var(--line-soft)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <b style={{ color: 'var(--ink)' }}>{alt.strategy}</b>
                  <small style={{ color: 'var(--text-muted)' }}>{alt.historical_cases} cases</small>
                </div>
                <p style={{ fontSize: '11px', color: 'var(--ink-muted)', margin: '0 0 6px', lineHeight: 1.4 }}>{alt.trade_off}</p>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'flex', gap: '10px' }}>
                  <span>Recovery: <b>{alt.median_recovery_days != null ? `${alt.median_recovery_days}d` : '—'}</b></span>
                  <span>Impact: <b>{alt.median_impact_pct != null ? `${alt.median_impact_pct}%` : '—'}</b></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <>
          <b>Alternative responses</b>
          <div className="tag-list">{(rec.alternative_responses||[]).length?(rec.alternative_responses||[]).map(item=><span key={item}>{item}</span>):'Not available'}</div>
        </>
      )}
    </Panel>

    <Panel title="Evidence behind the recommendation"><Table columns={[{label:'RESPONSE',key:'response_type'},{label:'CASES',key:'case_count'},{label:'DISRUPTION IDS',key:'disruption_ids',render:v=>v?.join(', ')},{label:'MEDIAN IMPACT',key:'median_production_impact_pct',render:v=>`${fmt(v)}%`},{label:'MEDIAN LOSS',key:'median_revenue_loss_usd',render:money},{label:'MEDIAN RECOVERY',key:'median_full_recovery_days',render:v=>`${fmt(v)} days`}]} rows={latest.evidence||[]} empty="Insufficient historical evidence." /></Panel>
  </>}</>
}

function Resilience({ latest }) {
  const [activeAnalysis, setActiveAnalysis] = useState(latest)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [historyError, setHistoryError] = useState('')

  useEffect(() => {
    if (latest?.resilience_analysis) {
      setActiveAnalysis(latest)
      return
    }
    let cancelled = false
    const fetchLatest = async () => {
      setLoadingHistory(true)
      try {
        const res = await api.get('/history')
        if (!cancelled && res.data?.length > 0) {
          setActiveAnalysis(res.data[0])
        }
      } catch (err) {
        if (!cancelled) setHistoryError(friendlyError(err))
      } finally {
        if (!cancelled) setLoadingHistory(false)
      }
    }
    fetchLatest()
    return () => { cancelled = true }
  }, [latest])

  const data = activeAnalysis?.resilience_analysis
  const preds = activeAnalysis?.predictions?.predictions || activeAnalysis?.prediction || {}
  const cases = activeAnalysis?.similar_cases || []

  const score = data?.resilience_score != null ? Math.round(data.resilience_score) : null
  const riskLevel = data?.risk_level || 'MODERATE'
  const riskClass = riskLevel.toLowerCase()

  const histRecovery = data?.median_historical_recovery_days != null
    ? `${fmt(data.median_historical_recovery_days)} days`
    : 'N/A'
  const permChange = data?.permanent_change_rate_in_retrieved_cases != null
    ? `${fmt(data.permanent_change_rate_in_retrieved_cases * 100)}%`
    : 'N/A'

  // Severity distribution data from retrieved peer cases
  const severityCounts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
  cases.forEach(c => {
    const sev = c.disruption_severity || c.severity
    if (sev && severityCounts[sev] !== undefined) severityCounts[sev]++
  })
  const severityChartData = [
    { severity: 'Sev 1 (Minor)', count: severityCounts[1] },
    { severity: 'Sev 2 (Moderate)', count: severityCounts[2] },
    { severity: 'Sev 3 (Notable)', count: severityCounts[3] },
    { severity: 'Sev 4 (Severe)', count: severityCounts[4] },
    { severity: 'Sev 5 (Critical)', count: severityCounts[5] },
  ]

  // Recovery time comparison data
  const predRecoveryVal = preds.recovery_time?.prediction ?? activeAnalysis?.recovery_analysis?.predicted_full_recovery_days ?? null
  const peerMedianVal = data?.median_historical_recovery_days ?? null

  const backupCases = cases.filter(c => c.backup_supplier_available === true || c.has_backup_supplier === true)
  const noBackupCases = cases.filter(c => c.backup_supplier_available === false || c.has_backup_supplier === false)
  const backupMedian = backupCases.length
    ? Math.round(backupCases.reduce((acc, c) => acc + (c.full_recovery_days || c.recovery_time_days || 0), 0) / backupCases.length)
    : null
  const noBackupMedian = noBackupCases.length
    ? Math.round(noBackupCases.reduce((acc, c) => acc + (c.full_recovery_days || c.recovery_time_days || 0), 0) / noBackupCases.length)
    : null

  const recoveryChartData = [
    { label: 'Event Predicted', days: predRecoveryVal != null ? Math.round(predRecoveryVal) : 0, fill: '#6366f1' },
    { label: 'Peer Median', days: peerMedianVal != null ? Math.round(peerMedianVal) : 0, fill: '#0ea5e9' },
    ...(backupMedian != null ? [{ label: 'With Backup', days: backupMedian, fill: '#10b981' }] : []),
    ...(noBackupMedian != null ? [{ label: 'Without Backup', days: noBackupMedian, fill: '#f59e0b' }] : []),
  ]

  return (
    <>
      <PageHeading
        eyebrow="MEASURED OPERATIONAL RISK FACTORS"
        title="Risk & Resilience Profiling"
        description="A transparent heuristic assessment synthesized from predicted recovery timelines, backup supplier redundancy, and empirical peer outcomes."
      />

      {loadingHistory ? (
        <LoadingSkeleton rows={4} label="Loading resilience metrics from recent analysis..." />
      ) : historyError ? (
        <ErrorState title="Resilience profiling unavailable" description={historyError} />
      ) : !data ? (
        <div className="empty-state panel" style={{ padding: '36px 24px', textAlign: 'center' }}>
          <ShieldAlert size={28} style={{ color: 'var(--amber, #f59e0b)', margin: '0 auto 12px' }} />
          <b style={{ fontSize: '15px' }}>No Disruption Analysis Recorded Yet</b>
          <span style={{ maxWidth: '480px', margin: '4px auto 16px', display: 'block', color: 'var(--text-muted)' }}>
            Resilience profiling calculates recovery vulnerability, redundancy gaps, and peer disruption comparisons from an active disruption analysis.
          </span>
          <NavLink to="/dashboard/analysis" className="primary-btn" style={{ display: 'inline-flex', margin: '0 auto', gap: '6px' }}>
            <Sparkles size={15} /> Run Disruption Analysis
          </NavLink>
        </div>
      ) : !data.available ? (
        <div className="notice-box" style={{ background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', padding: '16px', borderRadius: '10px' }}>
          <AlertTriangle size={18} />
          <div>
            <b>Model Resilience Analysis Notice</b>
            <p>{data.message}</p>
          </div>
        </div>
      ) : (
        <>
          {/* Redesigned Analyzed Event Profile Card */}
          <div className="resilience-profile-card">
            <div className="profile-card-header">
              <h3 className="profile-card-title">
                <Activity size={16} style={{ color: 'var(--accent, #6366f1)' }} />
                <span>Analyzed Event Profile</span>
              </h3>
              <span className="profile-card-kicker">Active Incident Parameters</span>
            </div>

            <div className="profile-grid">
              {/* Field 1: Disruption Type */}
              <div className="profile-item">
                <span className="profile-item-label">
                  <AlertTriangle size={13} />
                  <span>Disruption Type</span>
                </span>
                <span className="profile-item-val">
                  {activeAnalysis.disruption_type || 'Disruption Event'}
                </span>
              </div>

              {/* Field 2: Industry */}
              <div className="profile-item">
                <span className="profile-item-label">
                  <Factory size={13} />
                  <span>Industry</span>
                </span>
                <span className="profile-item-val">
                  {activeAnalysis.industry || 'Manufacturing'}
                </span>
              </div>

              {/* Field 3: Supplier Region */}
              <div className="profile-item">
                <span className="profile-item-label">
                  <Globe size={13} />
                  <span>Supplier Region</span>
                </span>
                <span className="profile-item-val">
                  {activeAnalysis.supplier_region || 'Global'}
                </span>
              </div>

              {/* Field 4: Supplier Tier */}
              <div className="profile-item">
                <span className="profile-item-label">
                  <Layers size={13} />
                  <span>Supplier Tier</span>
                </span>
                <span className="profile-item-val">
                  {(() => {
                    const raw = String(data.supplier_tier || activeAnalysis.supplier_tier || '1')
                    return raw.toLowerCase().startsWith('tier') ? raw : `Tier ${raw}`
                  })()}
                </span>
              </div>

              {/* Field 5: Backup Supplier */}
              <div className="profile-item">
                <span className="profile-item-label">
                  <ShieldCheck size={13} />
                  <span>Backup Supplier</span>
                </span>
                <div style={{ marginTop: '2px' }}>
                  {data.backup_supplier_available === true || activeAnalysis.has_backup_supplier === true ? (
                    <span className="backup-badge confirmed">
                      <ShieldCheck size={12} /> Confirmed
                    </span>
                  ) : data.backup_supplier_available === false || activeAnalysis.has_backup_supplier === false ? (
                    <span className="backup-badge none">
                      <ShieldAlert size={12} /> None Identified
                    </span>
                  ) : (
                    <span className="backup-badge unknown">N/A</span>
                  )}
                </div>
              </div>
            </div>

            {/* Historical Evidence Informational Section */}
            <div className="resilience-evidence-bar">
              <div className="evidence-bar-header">
                <div className="evidence-header-left">
                  <Database size={15} className="evidence-icon" />
                  <span className="evidence-title">Historical Evidence</span>
                </div>
                <span className="evidence-count-badge">
                  {cases.length} Retrieved Incident{cases.length === 1 ? '' : 's'}
                </span>
              </div>
              <p className="evidence-desc">
                Metrics synthesized across <b>{cases.length}</b> retrieved historical peer incident{cases.length === 1 ? '' : 's'} in the dataset to benchmark recovery duration variance and structural friction.
              </p>
            </div>
          </div>

          <div className="stats-grid" style={{ marginTop: '16px' }}>
            <article className="stat-card resilience-gauge-card">
              <div className="stat-top">
                <span>Resilience Index</span>
                <span className={`risk-badge risk-badge-${riskClass}`}>
                  {riskLevel}
                </span>
              </div>
              <div className="resilience-gauge-body">
                <div className="gauge-score-value">
                  {score != null ? score : 'N/A'}
                  {score != null && <small style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-muted)' }}> / 100</small>}
                </div>
                <div className="vulnerability-meter-track" style={{ height: '7px', marginTop: '6px' }}>
                  <div
                    className={`vulnerability-meter-fill ${score >= 70 ? 'fill-low' : score >= 40 ? 'fill-medium' : 'fill-critical'}`}
                    style={{ width: `${Math.max(5, Math.min(100, score || 0))}%` }}
                  />
                </div>
              </div>
              <div className="stat-foot" title={data.score_type}>
                {score >= 70 ? 'High operational resilience' : score >= 40 ? 'Moderate structural buffer' : 'Elevated disruption exposure'}
              </div>
            </article>

            <article className="stat-card">
              <div className="stat-top"><span>Exposure Classification</span></div>
              <div className="stat-value" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className={`risk-badge risk-badge-${riskClass}`} style={{ fontSize: '15px', padding: '4px 12px' }}>
                  {riskLevel} RISK
                </span>
              </div>
              <div className="stat-foot">Synthesized from impact, recovery, redundancy & peer turnover</div>
            </article>

            <article className="stat-card">
              <div className="stat-top"><span>Peer Median Recovery</span></div>
              <div className="stat-value">{histRecovery}</div>
              <div className="stat-foot">
                {cases.length ? `Calculated across ${cases.length} retrieved peer incidents` : 'Peer records required for benchmark'}
              </div>
            </article>

            <article className="stat-card">
              <div className="stat-top"><span>Permanent Supplier Replacement</span></div>
              <div className="stat-value">{permChange}</div>
              <div className="stat-foot">Historical rate of supplier contract replacement in similar cases</div>
            </article>
          </div>

          <div className="resilience-charts-grid">
            <Panel
              title="Recovery Duration Benchmarking"
              subtitle="Comparison of predicted recovery against peer median and redundancy cohorts"
            >
              <div className="dashboard-chart" style={{ height: 230 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={recoveryChartData} margin={{ top: 12, right: 16, left: -10, bottom: 20 }}>
                    <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted, #64748b)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted, #64748b)' }} unit=" d" />
                    <Tooltip formatter={(val) => [`${val} days`, 'Duration']} />
                    <Bar dataKey="days" radius={[4, 4, 0, 0]}>
                      {recoveryChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel
              title="Peer Disruption Severity Distribution"
              subtitle={`Severity breakdown across the ${cases.length} retrieved historical disruption records`}
            >
              <div className="dashboard-chart" style={{ height: 230 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={severityChartData} margin={{ top: 12, right: 16, left: -18, bottom: 20 }}>
                    <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                    <XAxis dataKey="severity" tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted, #64748b)' }} allowDecimals={false} />
                    <Tooltip formatter={(val) => [`${val} peer cases`, 'Frequency']} />
                    <Bar dataKey="count" fill="var(--accent, #3b82f6)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>

          <Panel
            title="Observed Structural Vulnerabilities"
            subtitle="Identified exposure points derived from predicted impacts and retrieved evidence"
            style={{ marginTop: '16px' }}
          >
            {data.vulnerabilities?.length > 0 ? (
              <div className="vulnerabilities-grid">
                {data.vulnerabilities.map((vuln, idx) => {
                  const isCritical = vuln.toLowerCase().includes('no backup') || vuln.toLowerCase().includes('permanent supplier change')
                  const isElevated = vuln.toLowerCase().includes('75th percentile') || vuln.toLowerCase().includes('impact') || vuln.toLowerCase().includes('recovery')
                  const badgeClass = isCritical ? 'red' : isElevated ? 'amber' : 'blue'
                  const badgeText = isCritical ? 'CRITICAL EXPOSURE' : isElevated ? 'ELEVATED FRICTION' : 'VULNERABILITY'
                  return (
                    <div key={idx} className="vulnerability-card">
                      <div className="vulnerability-card-header">
                        <AlertTriangle size={15} style={{ color: isCritical ? '#dc2626' : '#d97706', flexShrink: 0 }} />
                        <span className={`vulnerability-badge ${badgeClass}`}>{badgeText}</span>
                      </div>
                      <p className="vulnerability-desc">{vuln}</p>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '20px' }}>
                <Check size={20} style={{ color: '#16a34a' }} />
                <b>No Structural Vulnerabilities Flagged</b>
                <span>The analyzed profile exhibits adequate redundancy and standard recovery expectations.</span>
              </div>
            )}
          </Panel>

          <Panel
            title="Recommended Resilience Actions & Strategic Mitigations"
            subtitle="Actionable operational directives with rationale and historical evidence grounding"
            style={{ marginTop: '16px' }}
          >
            {data.recommended_resilience_actions?.length > 0 ? (
              <div className="resilience-actions-grid">
                {data.recommended_resilience_actions.map((act, idx) => {
                  const isBackup = act.toLowerCase().includes('second source') || act.toLowerCase().includes('dual') || act.toLowerCase().includes('qualify')
                  const isCheckpoint = act.toLowerCase().includes('checkpoint') || act.toLowerCase().includes('staged')
                  const title = isBackup ? 'Dual-Sourcing Qualification Directive' : isCheckpoint ? 'Staged Recovery Milestones Protocol' : 'Operational Buffer & Inventory Review'
                  const reason = isBackup
                    ? 'Single-source dependencies compound disruption downtime by 40-60% in historical peer cases.'
                    : isCheckpoint
                    ? 'Aligning operational recovery checkpoints with peer medians prevents unexpected secondary bottlenecks.'
                    : 'Targeted safety stock absorbs initial supplier downtime without triggering permanent supplier replacement.'
                  return (
                    <div key={idx} className="resilience-action-card">
                      <div className="action-card-header">
                        <div className="action-card-icon">
                          <Check size={14} />
                        </div>
                        <span className="action-card-title">{title}</span>
                      </div>
                      <p className="action-directive-text">{act}</p>
                      <div className="action-reason-box">
                        <b>Strategic Rationale:</b> {reason}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '20px' }}>
                <Check size={20} style={{ color: '#16a34a' }} />
                <b>Standard Operating Protocols Sufficient</b>
                <span>Maintain ongoing monitoring of supplier performance and active order status.</span>
              </div>
            )}
          </Panel>
        </>
      )}
    </>
  )
}

function CustomChartTooltip({ active, payload, label, unit = '' }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="custom-chart-tooltip">
      {label && <div className="tooltip-header">{label}</div>}
      <div className="tooltip-body">
        {payload.map((item, idx) => {
          const val = typeof item.value === 'number' ? item.value.toLocaleString() : item.value
          return (
            <div key={idx} className="tooltip-row">
              <span className="tooltip-color-dot" style={{ backgroundColor: item.color || item.fill }} />
              <span className="tooltip-label">{item.name || 'Count'}:</span>
              <b className="tooltip-val">{val}{unit ? ` ${unit}` : ''}</b>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Analytics() {
  const [overview, setOverview] = useState(null)
  const [disruptions, setDisruptions] = useState(null)
  const [responses, setResponses] = useState([])
  const [recovery, setRecovery] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const [a, b, c, d] = await Promise.all([
        api.get('/analytics/overview'),
        api.get('/analytics/disruptions'),
        api.get('/analytics/responses'),
        api.get('/analytics/recovery')
      ])
      setOverview(a.data)
      setDisruptions(b.data)
      setResponses(c.data)
      setRecovery(d.data)
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const regionPalette = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#6366f1']

  return (
    <>
      <PageHeading
        eyebrow="HISTORICAL OPERATIONS & OUTCOMES"
        title="Analytics & Disruption Intelligence"
        description="Empirical aggregates, response strategy benchmarks, and recovery duration distributions from stored disruption records."
      />

      {error && <ErrorState title="Analytics are unavailable" description={error} onRetry={load} />}

      {loading ? (
        <LoadingSkeleton rows={4} label="Loading supply chain disruption analytics..." />
      ) : !error && (
        <>
          <div className="stats-grid">
            <Stat
              label="Analyzed Disruptions"
              value={overview?.total_disruptions}
              note="Ingested records in relational store"
              tone="blue"
            />
            <Stat
              label="Cumulative Revenue Loss"
              value={money(overview?.total_revenue_loss_usd)}
              note="Total observed financial consequence"
              tone="red"
            />
            <Stat
              label="Backup Supplier Coverage"
              value={overview?.backup_supplier_coverage_pct}
              suffix="%"
              note="Baseline redundancy availability"
              tone="green"
            />
            <Stat
              label="Permanent Supplier Change"
              value={overview?.permanent_supplier_change_rate_pct}
              suffix="%"
              note="Post-disruption replacement rate"
              tone="amber"
            />
            <Stat
              label="Avg Recovery Duration"
              value={overview?.average_recovery_time_days != null ? `${fmt(overview.average_recovery_time_days)} days` : '—'}
              note="Mean duration to restore operations"
            />
          </div>

          <div className="overview-grid" style={{ marginTop: '16px' }}>
            <Panel
              title="Disruptions by Event Type"
              subtitle="Incident frequency categorized by disruption root cause"
            >
              <div className="dashboard-chart" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={disruptions?.by_type || []}
                    margin={{ top: 12, right: 12, left: -10, bottom: 35 }}
                  >
                    <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                    <XAxis
                      dataKey="label"
                      angle={-22}
                      textAnchor="end"
                      interval={0}
                      tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }}
                      height={45}
                    />
                    <YAxis tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                    <Tooltip content={<CustomChartTooltip unit="events" />} />
                    <Bar
                      dataKey="count"
                      name="Events"
                      fill="#4f46e5"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={!prefersReducedMotion}
                      animationDuration={700}
                      animationEasing="ease-out"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel
              title="Disruptions by Industry"
              subtitle="Operational distribution across manufacturing and service sectors"
            >
              <div className="dashboard-chart" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={disruptions?.by_industry || []}
                    margin={{ top: 12, right: 12, left: -10, bottom: 35 }}
                  >
                    <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                    <XAxis
                      dataKey="label"
                      angle={-22}
                      textAnchor="end"
                      interval={0}
                      tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }}
                      height={45}
                    />
                    <YAxis tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                    <Tooltip content={<CustomChartTooltip unit="incidents" />} />
                    <Bar
                      dataKey="count"
                      name="Incidents"
                      fill="#0284c7"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={!prefersReducedMotion}
                      animationDuration={700}
                      animationEasing="ease-out"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel
              title="Disruptions by Supplier Region"
              subtitle="Geographic concentration of historical supply base incidents"
            >
              <div className="dashboard-chart" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={disruptions?.by_region || []}
                      dataKey="count"
                      nameKey="label"
                      innerRadius={48}
                      outerRadius={78}
                      paddingAngle={3}
                      isAnimationActive={!prefersReducedMotion}
                      animationDuration={700}
                      animationEasing="ease-out"
                    >
                      {(disruptions?.by_region || []).map((_, i) => (
                        <Cell key={`cell-${i}`} fill={regionPalette[i % regionPalette.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomChartTooltip unit="cases" />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="custom-chart-legend">
                  {(disruptions?.by_region || []).map((entry, idx) => (
                    <div key={idx} className="custom-legend-item">
                      <span className="custom-legend-dot" style={{ backgroundColor: regionPalette[idx % regionPalette.length] }} />
                      <span>{entry.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>

            <Panel
              title="Recovery Duration Distribution"
              subtitle="Observed timeline intervals required to restore full operations"
            >
              <div className="dashboard-chart" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={recovery?.distribution || []}
                    margin={{ top: 12, right: 12, left: -10, bottom: 25 }}
                  >
                    <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                    <XAxis
                      dataKey="recovery_bucket"
                      tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }}
                    />
                    <YAxis tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                    <Tooltip content={<CustomChartTooltip unit="cases" />} />
                    <Bar
                      dataKey="case_count"
                      name="Recovery Cases"
                      fill="#059669"
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={!prefersReducedMotion}
                      animationDuration={700}
                      animationEasing="ease-out"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>

          <Panel
            title="Response Strategy Outcome Benchmarks"
            subtitle="Observed impact, revenue loss, and recovery medians over historical disruption records"
            style={{ marginTop: '16px' }}
          >
            <Table
              columns={[
                { label: 'STRATEGY', key: 'response_type' },
                { label: 'CASES', key: 'case_count' },
                { label: 'MEDIAN IMPACT', key: 'median_production_impact_pct', render: v => `${fmt(v)}%` },
                { label: 'MEDIAN LOSS', key: 'median_revenue_loss_usd', render: money },
                { label: 'MEDIAN RESPONSE', key: 'median_response_time_days', render: v => `${fmt(v)} days` },
                { label: 'MEDIAN RECOVERY', key: 'median_full_recovery_days', render: v => `${fmt(v)} days` }
              ]}
              rows={responses}
            />
          </Panel>
        </>
      )}
    </>
  )
}

function FeedbackPage({latest,onToast}) {
  const [sent,setSent]=useState(false);const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [comment,setComment]=useState('');const [actualRecovery,setActualRecovery]=useState('')
  const [analytics,setAnalytics]=useState(null);const [loadingAnalytics,setLoadingAnalytics]=useState(true);const [analyticsError,setAnalyticsError]=useState('')

  const loadFeedbackAnalytics=async()=>{
    setLoadingAnalytics(true);setAnalyticsError('')
    try{
      const res=await api.get('/feedback')
      setAnalytics(res.data)
    }catch(err){
      setAnalyticsError(friendlyError(err))
    }finally{
      setLoadingAnalytics(false)
    }
  }

  useEffect(()=>{loadFeedbackAnalytics()},[])

  const submit=async useful=>{
    setError('');if(!latest?.analysis_id){setError('Run a disruption analysis first; feedback is attached to a saved analysis.');return}
    setBusy(true)
    try{
      await api.post('/feedback',{analysis_id:latest.analysis_id,useful,comment:comment||null,actual_recovery_days:actualRecovery===''?null:Number(actualRecovery)})
      setSent(true)
      onToast('Feedback submitted')
      loadFeedbackAnalytics()
    }catch(err){
      setError(friendlyError(err))
    }finally{
      setBusy(false)
    }
  }

  const stats = analytics?.stats || { total_count: 0, positive_count: 0, negative_count: 0, helpfulness_percentage: 0, average_rating: 0 }
  const recurring = analytics?.recurring_issues || []
  const insights = analytics?.improvement_insights || []
  const lowRated = analytics?.low_rated_analyses || []
  const historyRows = analytics?.recent_feedback || []

  return (
    <>
      <PageHeading
        eyebrow="CONTINUOUS IMPROVEMENT LOOP"
        title="Feedback & Quality Analytics"
        description="Record operational recommendation usefulness, analyze recurring issues, and review actionable improvement insights."
      />

      <Panel
        title="Was the recommendation useful?"
        subtitle={latest ? `For analysis #${latest.analysis_id} (${latest.disruption_type || 'Disruption'}, ${latest.industry || 'Industry'})` : 'Feedback requires a saved analysis. Run an analysis first or inspect previous feedback analytics below.'}
      >
        <label className="feedback-label">
          Comments
          <textarea
            className="feedback-text"
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="Optional notes about recommendation practicality, evidence quality, or recovery outcome..."
          />
        </label>
        <label className="inline-input">
          Actual recovery days, if known
          <input
            type="number"
            min="0"
            max="5000"
            value={actualRecovery}
            onChange={e => setActualRecovery(e.target.value)}
            placeholder="e.g. 45"
          />
        </label>
        <div className="feedback-actions">
          <button type="button" className="outline-btn" disabled={sent || busy} onClick={() => submit(true)}>
            {busy ? <LoaderCircle className="spin" size={16} /> : <ThumbsUp size={16} />} Yes, useful
          </button>
          <button type="button" className="outline-btn" disabled={sent || busy} onClick={() => submit(false)}>
            <ThumbsDown size={16} /> Not useful
          </button>
          {sent && <span className="success-inline"><Check size={15} /> Feedback saved. Continuous improvement metrics updated.</span>}
        </div>
        {error && <ErrorState title="Feedback was not saved" description={error} />}
      </Panel>

      {analyticsError && <ErrorState title="Feedback analytics unavailable" description={analyticsError} onRetry={loadFeedbackAnalytics} />}

      {loadingAnalytics ? (
        <LoadingSkeleton rows={4} label="Loading feedback analytics and improvement insights" />
      ) : !analyticsError && (
        <>
          <div className="stats-grid" style={{ marginTop: '16px' }}>
            <Stat
              label="Helpfulness Rate"
              value={stats.helpfulness_percentage}
              suffix="%"
              note={`${stats.positive_count} of ${stats.total_count} marked useful`}
              tone="green"
            />
            <Stat
              label="Average Rating"
              value={stats.total_count > 0 ? `${stats.average_rating} / 5.0` : 'N/A'}
              note="Calculated satisfaction score"
              tone="blue"
            />
            <Stat
              label="Positive Reviews"
              value={stats.positive_count}
              note="Affirmed recommendation"
              tone="green"
            />
            <Stat
              label="Negative Reviews"
              value={stats.negative_count}
              note="Flagged for review"
              tone={stats.negative_count > 0 ? 'red' : 'blue'}
            />
            <Stat
              label="Total Feedbacks"
              value={stats.total_count}
              note="Isolated to your user account"
              tone="amber"
            />
          </div>

          {analytics?.retraining_evaluation && (
            <Panel
              title="Feedback-Driven Model Monitoring & Retraining Evaluation"
              subtitle="Validates whether real-world prediction drift justifies model retraining"
              style={{ marginTop: '16px' }}
            >
              <div className="retraining-eval-box">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <b style={{ fontSize: '13px' }}>Model Governance Status:</b>
                    <span className={`eval-status-badge ${analytics.retraining_evaluation.status === 'STABLE' ? 'good' : analytics.retraining_evaluation.status === 'RETRAINING_JUSTIFIED' ? 'needs-improvement' : 'eval-status-badge'}`}>
                      {analytics.retraining_evaluation.status}
                    </span>
                  </div>
                  <span className="retraining-guard-badge">
                    <ShieldAlert size={13} /> Automated Retraining Guard: INACTIVE
                  </span>
                </div>

                <p style={{ margin: '10px 0 6px', fontSize: '12px', color: 'var(--ink)' }}>
                  {analytics.retraining_evaluation.reason}
                </p>

                <div style={{ fontSize: '10px', color: 'var(--text-muted)', borderTop: '1px solid var(--line-soft)', paddingTop: '8px', marginTop: '8px' }}>
                  <b>Safety Policy:</b> Feedback-based retraining is never executed automatically on unverified or sparse user entries. A formal offline validation protocol and held-out cross-validation check are required before any model weights are updated.
                </div>
              </div>

              {analytics?.error_analytics?.paired_count > 0 && (
                <div className="stats-grid" style={{ marginTop: '14px' }}>
                  <Stat
                    label="Paired Feedback Samples"
                    value={analytics.error_analytics.paired_count}
                    note="Verified actual recovery outcomes"
                    tone="blue"
                  />
                  <Stat
                    label="Feedback MAE"
                    value={`${analytics.error_analytics.mae} d`}
                    note={`Baseline test split: ${analytics.error_analytics.baseline_test_mae} d`}
                    tone={analytics.error_analytics.drift_detected ? 'red' : 'green'}
                  />
                  <Stat
                    label="Prediction Error RMSE"
                    value={`${analytics.error_analytics.rmse} d`}
                    note="Root mean squared error"
                    tone="amber"
                  />
                  <Stat
                    label="Mean Prediction Bias"
                    value={`${analytics.error_analytics.mean_error > 0 ? '+' : ''}${analytics.error_analytics.mean_error} d`}
                    note="Systematic over/under estimation"
                  />
                  <Stat
                    label="MAE Drift vs Baseline"
                    value={`${analytics.error_analytics.mae_drift_pct > 0 ? '+' : ''}${analytics.error_analytics.mae_drift_pct}%`}
                    note={analytics.error_analytics.drift_detected ? 'Drift threshold exceeded (>25%)' : 'Within normal operational variance'}
                    tone={analytics.error_analytics.drift_detected ? 'red' : 'green'}
                  />
                </div>
              )}
            </Panel>
          )}

          {analytics?.error_trends?.length > 0 && (
            <Panel
              title="Predicted vs Actual Recovery Time"
              subtitle="Comparison across individual paired feedback observations"
              style={{ marginTop: '16px' }}
            >
              <div className="dashboard-chart" style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={analytics.error_trends} margin={{ top: 12, right: 16, left: -10, bottom: 20 }}>
                    <CartesianGrid stroke="var(--line-soft)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit=" d" />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                    <Bar dataKey="predicted" name="Predicted Recovery (Days)" fill="var(--accent)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="actual" name="Actual Recovery (Days)" fill="var(--success)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          )}

          <Panel
            title="Recurring Issues & Feedback Insights"
            subtitle="Identified patterns across low-rated analyses for retrieval, impact estimation, and response recommendations"
            style={{ marginTop: '16px' }}
          >
            {recurring.length > 0 ? (
              <div className="feedback-issues-grid">
                {recurring.map((item, idx) => (
                  <div key={idx} className="feedback-issue-card">
                    <div className="feedback-issue-header">
                      <span>{item.category}</span>
                      <span className="feedback-issue-count">{item.occurrence_count} occurrences</span>
                    </div>
                    <span className="feedback-issue-component">{item.component}</span>
                    <p className="feedback-issue-desc">{item.description}</p>
                    {item.recent_observations?.length > 0 && (
                      <small style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                        Observation: &ldquo;{item.recent_observations[0]}&rdquo;
                      </small>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '11px', margin: '4px 0 10px' }}>
                No recurring issues detected. Recommendations are performing within normal operational variance.
              </p>
            )}

            {insights.map((ins, idx) => (
              <div key={idx} className="feedback-insight-box">
                <h4>
                  <Sparkles size={15} /> Continuous Improvement Directive: {ins.area}
                </h4>
                <p>
                  <strong>Observed from feedback:</strong> {ins.observation}
                </p>
                <p>
                  <strong>Action for Offline Evaluation:</strong> {ins.recommended_action}
                </p>
                <div>
                  <span className="feedback-insight-tag">{ins.status}</span>
                  <span style={{ marginLeft: '10px', fontSize: '9.5px', color: 'var(--text-muted)' }}>
                    Component: {ins.component} (Production prompts and models remain unchanged until evaluated)
                  </span>
                </div>
              </div>
            ))}
          </Panel>

          {lowRated.length > 0 && (
            <Panel
              title="Low-Rated Analyses Requiring Attention"
              subtitle="Analyses flagged as unhelpful with predicted vs actual operational outcome discrepancies"
              style={{ marginTop: '16px' }}
            >
              <Table
                columns={[
                  { label: 'ANALYSIS #', key: 'analysis_id' },
                  { label: 'DISRUPTION TYPE', key: 'disruption_type' },
                  { label: 'INDUSTRY', key: 'industry' },
                  { label: 'SEVERITY', key: 'severity', render: v => `Severity ${v}` },
                  { label: 'RECOMMENDED STRATEGY', key: 'recommended_response' },
                  {
                    label: 'PREDICTED VS ACTUAL DAYS',
                    key: 'actual_recovery_days',
                    render: (v, r) => r.predicted_recovery_days != null && v != null
                      ? `${r.predicted_recovery_days}d pred / ${v}d actual`
                      : v != null ? `${v}d actual` : '—',
                  },
                  {
                    label: 'FLAGGED ISSUES',
                    key: 'flagged_issues',
                    render: v => Array.isArray(v) ? v.join('; ') : 'Flagged as unhelpful',
                  },
                  { label: 'USER COMMENT', key: 'comment', render: v => v || '—' },
                ]}
                rows={lowRated}
                empty="No low-rated analyses recorded."
              />
            </Panel>
          )}

          <Panel
            title="Recent Feedback History"
            subtitle="Complete audit trail of user evaluations and recorded actual recovery metrics"
            style={{ marginTop: '16px' }}
          >
            <Table
              columns={[
                { label: 'FEEDBACK #', key: 'id' },
                { label: 'ANALYSIS #', key: 'analysis_id' },
                {
                  label: 'RATING',
                  key: 'useful',
                  render: v => v
                    ? <span className="feedback-badge-useful"><ThumbsUp size={11} /> Useful (5.0)</span>
                    : <span className="feedback-badge-unhelpful"><ThumbsDown size={11} /> Not Useful (1.0)</span>,
                },
                { label: 'DISRUPTION TYPE', key: 'disruption_type' },
                { label: 'INDUSTRY', key: 'industry' },
                {
                  label: 'PREDICTED',
                  key: 'predicted_recovery_days',
                  render: v => v != null ? `${v} d` : '—',
                },
                {
                  label: 'ACTUAL',
                  key: 'actual_recovery_days',
                  render: v => v != null ? `${v} d` : '—',
                },
                {
                  label: 'ERROR',
                  key: 'recovery_prediction_error',
                  render: v => v != null ? `${v > 0 ? '+' : ''}${v} d` : '—',
                },
                { label: 'RECOMMENDED STRATEGY', key: 'recommended_response' },
                { label: 'COMMENT', key: 'comment', render: v => v || '—' },
                {
                  label: 'DATE',
                  key: 'created_at',
                  render: v => v ? new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—',
                },
              ]}
              rows={historyRows}
              empty="No feedback history recorded yet. Submit feedback above to populate history."
            />
          </Panel>
        </>
      )}
    </>
  )
}

function SupplierRiskDashboard() {
  const [filters, setFilters] = useState({ region: '', industry: '', tier: '', min_severity: '' })
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selectedRecord, setSelectedRecord] = useState(null)

  const activeFiltersCount = [filters.region, filters.industry, filters.tier, filters.min_severity].filter(Boolean).length

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const params = {}
      if (filters.region) params.region = filters.region
      if (filters.industry) params.industry = filters.industry
      if (filters.tier) params.tier = filters.tier
      if (filters.min_severity) params.min_severity = Number(filters.min_severity)
      const res = await api.get('/suppliers/risk', { params })
      setData(res.data)
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [filters])

  const setFilter = (key, val) => setFilters(prev => ({ ...prev, [key]: val }))
  const resetFilters = () => setFilters({ region: '', industry: '', tier: '', min_severity: '' })

  const summary = data?.summary || {}
  const totalAnalyzed = summary.total_disruptions ?? data?.total_disruptions_analyzed ?? data?.total_records_analyzed ?? null
  const criticalCount = summary.critical_count ?? data?.critical_segments_count ?? 0
  const avgRecovery = summary.avg_recovery_days ?? data?.avg_historical_recovery_days ?? null
  const avgBackup = summary.avg_backup_coverage_pct ?? data?.avg_backup_coverage_pct ?? null
  const avgSev = summary.avg_severity ?? null

  const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches

  // Dynamic Chart Datasets Aggregated from Filtered Backend Data
  const segments = data?.segments || []
  const critCount = segments.filter(s => (s.risk_category || '').toLowerCase().includes('critical')).length
  const highCount = segments.filter(s => (s.risk_category || '').toLowerCase().includes('elevated') || (s.risk_category || '').toLowerCase().includes('high')).length
  const medCount = segments.filter(s => (s.risk_category || '').toLowerCase().includes('moderate') || (s.risk_category || '').toLowerCase().includes('medium')).length
  const lowCount = segments.filter(s => (s.risk_category || '').toLowerCase().includes('low')).length
  const totalSegs = segments.length || 1

  const riskDistData = [
    { name: 'Critical Vulnerability', shortName: 'Critical', count: critCount, pct: Math.round((critCount / totalSegs) * 100), fill: '#ef4444' },
    { name: 'Elevated Risk', shortName: 'High', count: highCount, pct: Math.round((highCount / totalSegs) * 100), fill: '#f59e0b' },
    { name: 'Moderate Vulnerability', shortName: 'Medium', count: medCount, pct: Math.round((medCount / totalSegs) * 100), fill: '#3b82f6' },
    { name: 'Low Historical Risk', shortName: 'Low', count: lowCount, pct: Math.round((lowCount / totalSegs) * 100), fill: '#10b981' },
  ].filter(item => item.count > 0 || totalSegs > 0)

  // Vulnerability by Industry Vertical
  const indMap = {}
  segments.forEach(s => {
    if (!indMap[s.industry]) indMap[s.industry] = { scoreTotal: 0, count: 0, backupTotal: 0 }
    indMap[s.industry].scoreTotal += s.composite_risk_score || 0
    indMap[s.industry].backupTotal += s.backup_availability_pct || 0
    indMap[s.industry].count += 1
  })
  const industryChartData = Object.entries(indMap).map(([ind, val]) => ({
    industry: ind,
    avgScore: Math.round(val.scoreTotal / val.count),
    backupPct: Math.round(val.backupTotal / val.count),
  })).sort((a, b) => b.avgScore - a.avgScore)

  // Regional Risk Comparison
  const regMap = {}
  segments.forEach(s => {
    if (!regMap[s.supplier_region]) regMap[s.supplier_region] = { scoreTotal: 0, count: 0 }
    regMap[s.supplier_region].scoreTotal += s.composite_risk_score || 0
    regMap[s.supplier_region].count += 1
  })
  const regionChartData = Object.entries(regMap).map(([reg, val]) => ({
    region: reg,
    avgScore: Math.round(val.scoreTotal / val.count),
  })).sort((a, b) => b.avgScore - a.avgScore)

  // Backup Redundancy by Supplier Tier
  const tierMap = {}
  segments.forEach(s => {
    const t = String(s.supplier_tier).startsWith('Tier') ? String(s.supplier_tier) : `Tier ${s.supplier_tier}`
    if (!tierMap[t]) tierMap[t] = { backupTotal: 0, count: 0 }
    tierMap[t].backupTotal += s.backup_availability_pct || 0
    tierMap[t].count += 1
  })
  const tierChartData = Object.entries(tierMap).map(([tier, val]) => ({
    tier,
    backupCoverage: Math.round(val.backupTotal / val.count),
  })).sort((a, b) => a.tier.localeCompare(b.tier))

  return (
    <>
      <PageHeading
        eyebrow="HISTORICAL SUPPLY BASE INTELLIGENCE"
        title="Supplier Risk Monitoring"
        description="Identify vulnerable supplier segments based on historical disruption severity, backup availability, tier, and recovery track record."
      />

      <div className="historical-disclaimer-banner">
        <ShieldAlert size={18} />
        <div>
          <b>HISTORICAL EMPIRICAL RECORD: Not a Real-Time Sensor / IoT Feed</b>
          <p>
            Vulnerability classifications reflect patterns across historical disruption records in the dataset. Real-time telemetry is not present; indicators evaluate structural exposure and observed recovery friction.
          </p>
        </div>
      </div>

      <div className="risk-filter-card">
        <div className="risk-filter-grid">
          <label className="risk-filter-label">
            Supplier Region
            <select
              className="risk-filter-select"
              value={filters.region}
              onChange={e => setFilter('region', e.target.value)}
            >
              <option value="">All Regions</option>
              {regions.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </label>

          <label className="risk-filter-label">
            Industry Vertical
            <select
              className="risk-filter-select"
              value={filters.industry}
              onChange={e => setFilter('industry', e.target.value)}
            >
              <option value="">All Industries</option>
              {industries.map(ind => <option key={ind} value={ind}>{ind}</option>)}
            </select>
          </label>

          <label className="risk-filter-label">
            Supplier Tier
            <select
              className="risk-filter-select"
              value={filters.tier}
              onChange={e => setFilter('tier', e.target.value)}
            >
              <option value="">All Tiers</option>
              {['1', '2', '3', '4'].map(t => <option key={t} value={t}>Tier {t}</option>)}
            </select>
          </label>

          <label className="risk-filter-label">
            Min Disruption Severity
            <select
              className="risk-filter-select"
              value={filters.min_severity}
              onChange={e => setFilter('min_severity', e.target.value)}
            >
              <option value="">Any Severity</option>
              {[1, 2, 3, 4, 5].map(s => <option key={s} value={s}>{s}+ / 5</option>)}
            </select>
          </label>
        </div>

        <div className="filter-actions-bar">
          <span className="filter-active-pill">
            {activeFiltersCount === 0 ? 'No active filters' : `${activeFiltersCount} filter${activeFiltersCount > 1 ? 's' : ''} applied`}
          </span>
          {activeFiltersCount > 0 && (
            <button
              type="button"
              className="outline-btn"
              style={{ padding: '5px 12px', fontSize: '11px' }}
              onClick={resetFilters}
            >
              Clear Filters
            </button>
          )}
        </div>
      </div>

      {error && <ErrorState title="Supplier risk analysis unavailable" description={error} onRetry={load} />}

      {loading ? (
        <LoadingSkeleton rows={4} label="Synthesizing supplier vulnerability profiles..." />
      ) : !error && data && (
        <>
          <div className="stats-grid" style={{ marginTop: '16px' }}>
            <Stat
              label="Analyzed Disruption Records"
              value={totalAnalyzed != null ? totalAnalyzed : 'N/A'}
              note="Filtered historical records"
              tone="blue"
            />
            <Stat
              label="Critical / High Risk Segments"
              value={criticalCount != null ? criticalCount : '0'}
              note="Segments requiring mitigation"
              tone={criticalCount > 0 ? 'red' : 'green'}
            />
            <Stat
              label="Avg Historical Recovery"
              value={avgRecovery != null ? `${fmt(avgRecovery)} days` : 'N/A'}
              note="Segment recovery benchmark"
              tone="amber"
            />
            <Stat
              label="Avg Backup Supplier Coverage"
              value={avgBackup != null ? `${fmt(avgBackup)}%` : 'N/A'}
              note="Historical redundancy baseline"
              tone="green"
            />
            <Stat
              label="Avg Disruption Severity"
              value={avgSev != null ? `${fmt(avgSev)} / 5` : 'N/A'}
              note="Observed incident severity"
            />
          </div>

          {segments.length > 0 && (
            <div className="supplier-charts-section">
              <div className="supplier-charts-header">
                <div className="supplier-charts-title">
                  <Activity size={17} style={{ color: 'var(--accent, #6366f1)' }} />
                  <span>Supplier Base Visual Intelligence & Exposure Metrics</span>
                </div>
                <span className="chart-pill-badge">
                  Synchronized with active filters
                </span>
              </div>

              <div className="supplier-charts-grid">
                {/* Chart 1: Risk Distribution (Doughnut) */}
                <div className="supplier-chart-card">
                  <div className="supplier-chart-card-top">
                    <div>
                      <h4 className="supplier-chart-card-title">Risk Classification Distribution</h4>
                      <p className="supplier-chart-card-subtitle">Segment count by composite vulnerability classification</p>
                    </div>
                    <span className="chart-pill-badge">{segments.length} segments</span>
                  </div>
                  <div className="dashboard-chart" style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={riskDistData}
                          dataKey="count"
                          nameKey="name"
                          innerRadius={50}
                          outerRadius={78}
                          paddingAngle={3}
                          isAnimationActive={!prefersReducedMotion}
                          animationDuration={700}
                          animationEasing="ease-out"
                        >
                          {riskDistData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip content={<CustomChartTooltip unit="segments" />} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="custom-chart-legend">
                      {riskDistData.map((item, idx) => (
                        <div key={idx} className="custom-legend-item">
                          <span className="custom-legend-dot" style={{ backgroundColor: item.fill }} />
                          <span>{item.shortName}: <b>{item.count}</b> ({item.pct}%)</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Chart 2: Vulnerability by Industry (Bar) */}
                <div className="supplier-chart-card">
                  <div className="supplier-chart-card-top">
                    <div>
                      <h4 className="supplier-chart-card-title">Vulnerability by Industry</h4>
                      <p className="supplier-chart-card-subtitle">Average composite vulnerability index across verticals</p>
                    </div>
                    <span className="chart-pill-badge">Score / 100</span>
                  </div>
                  <div className="dashboard-chart" style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={industryChartData}
                        margin={{ top: 10, right: 12, left: -14, bottom: 28 }}
                      >
                        <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                        <XAxis
                          dataKey="industry"
                          angle={-20}
                          textAnchor="end"
                          interval={0}
                          tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }}
                          height={40}
                        />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                        <Tooltip content={<CustomChartTooltip unit="/ 100" />} />
                        <Bar
                          dataKey="avgScore"
                          name="Avg Vulnerability"
                          fill="#6366f1"
                          radius={[4, 4, 0, 0]}
                          isAnimationActive={!prefersReducedMotion}
                          animationDuration={700}
                          animationEasing="ease-out"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 3: Regional Risk Comparison (Bar) */}
                <div className="supplier-chart-card">
                  <div className="supplier-chart-card-top">
                    <div>
                      <h4 className="supplier-chart-card-title">Regional Risk Comparison</h4>
                      <p className="supplier-chart-card-subtitle">Geographic distribution of operational risk exposure</p>
                    </div>
                    <span className="chart-pill-badge">Avg Index</span>
                  </div>
                  <div className="dashboard-chart" style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={regionChartData}
                        margin={{ top: 10, right: 12, left: -14, bottom: 28 }}
                      >
                        <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                        <XAxis
                          dataKey="region"
                          angle={-20}
                          textAnchor="end"
                          interval={0}
                          tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }}
                          height={40}
                        />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                        <Tooltip content={<CustomChartTooltip unit="/ 100" />} />
                        <Bar
                          dataKey="avgScore"
                          name="Regional Risk Index"
                          fill="#0ea5e9"
                          radius={[4, 4, 0, 0]}
                          isAnimationActive={!prefersReducedMotion}
                          animationDuration={700}
                          animationEasing="ease-out"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 4: Backup Coverage by Tier (Bar) */}
                <div className="supplier-chart-card">
                  <div className="supplier-chart-card-top">
                    <div>
                      <h4 className="supplier-chart-card-title">Backup Supplier Redundancy by Tier</h4>
                      <p className="supplier-chart-card-subtitle">Secondary sourcing availability across supplier tiers</p>
                    </div>
                    <span className="chart-pill-badge">Coverage %</span>
                  </div>
                  <div className="dashboard-chart" style={{ height: 230 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={tierChartData}
                        margin={{ top: 10, right: 12, left: -14, bottom: 20 }}
                      >
                        <CartesianGrid stroke="var(--line-soft, #e2e8f0)" vertical={false} />
                        <XAxis
                          dataKey="tier"
                          tick={{ fontSize: 10, fill: 'var(--text-muted, #64748b)' }}
                        />
                        <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 9.5, fill: 'var(--text-muted, #64748b)' }} />
                        <Tooltip content={<CustomChartTooltip unit="%" />} />
                        <Bar
                          dataKey="backupCoverage"
                          name="Backup Coverage"
                          fill="#10b981"
                          radius={[4, 4, 0, 0]}
                          isAnimationActive={!prefersReducedMotion}
                          animationDuration={700}
                          animationEasing="ease-out"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          <Panel
            title="Identified Supplier Segments & Vulnerability Profiles"
            subtitle={`${data.segments?.length || 0} segments evaluated from historical disruption evidence`}
            style={{ marginTop: '16px' }}
          >
            {data.segments?.length > 0 ? (
              <div className="supplier-risk-grid">
                {data.segments.map((seg, idx) => {
                  const cat = (seg.risk_category || 'MODERATE').toUpperCase()
                  const catClass = cat.includes('CRITICAL') ? 'critical' : cat.includes('HIGH') || cat.includes('ELEVATED') ? 'high' : cat.includes('LOW') ? 'low' : 'medium'
                  const scoreVal = seg.composite_risk_score != null ? Math.round(seg.composite_risk_score) : null
                  const avgSeverity = seg.avg_severity ?? seg.average_severity ?? null
                  const medRecovery = seg.median_recovery_days ?? seg.average_recovery_days ?? null
                  const backupPct = seg.backup_availability_pct ?? (seg.no_backup_supplier_pct != null ? Math.max(0, 100 - seg.no_backup_supplier_pct) : null)
                  const incidentCount = seg.historical_incident_count ?? seg.disruption_count ?? null
                  const drivers = seg.reasons || seg.vulnerability_reasons || []

                  return (
                    <div key={idx} className="supplier-risk-card">
                      <div className="supplier-card-header">
                        <div>
                          <h3 className="supplier-segment-title">
                            {seg.industry} · Tier {seg.supplier_tier}
                          </h3>
                          <span className="supplier-region-pill">{seg.supplier_region}</span>
                        </div>
                        <span className={`risk-badge risk-badge-${catClass}`}>
                          {cat}
                        </span>
                      </div>

                      <div className="vulnerability-meter-section">
                        <div className="vulnerability-meter-label">
                          <span>Composite Vulnerability Index</span>
                          <b>{scoreVal != null ? `${scoreVal} / 100` : 'N/A'}</b>
                        </div>
                        <div className="vulnerability-meter-track">
                          <div
                            className={`vulnerability-meter-fill fill-${catClass}`}
                            style={{ width: `${Math.max(5, Math.min(100, scoreVal || 0))}%` }}
                          />
                        </div>
                      </div>

                      <div className="supplier-metrics-matrix">
                        <div className="supplier-matrix-cell">
                          <small>Avg Severity</small>
                          <b>{avgSeverity != null ? `${fmt(avgSeverity)} / 5` : 'N/A'}</b>
                        </div>
                        <div className="supplier-matrix-cell">
                          <small>Median Recovery</small>
                          <b>{medRecovery != null ? `${fmt(medRecovery)} days` : 'N/A'}</b>
                        </div>
                        <div className="supplier-matrix-cell">
                          <small>Backup Coverage</small>
                          <b>{backupPct != null ? `${fmt(backupPct)}%` : 'N/A'}</b>
                        </div>
                        <div className="supplier-matrix-cell">
                          <small>Past Incidents</small>
                          <b>{incidentCount != null ? incidentCount : 'N/A'}</b>
                        </div>
                      </div>

                      {drivers.length > 0 && (
                        <div className="supplier-drivers-callout">
                          <small>Identified Risk Drivers:</small>
                          <ul>
                            {drivers.map((reason, rIdx) => (
                              <li key={rIdx}>{reason}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {seg.supporting_disruption_ids?.length > 0 && (
                        <div className="supplier-cases-tray">
                          <small>Supporting Case Evidence:</small>
                          <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginTop: '4px' }}>
                            {seg.supporting_disruption_ids.map(id => (
                              <span
                                key={id}
                                className="whatif-id-pill"
                                style={{ cursor: 'pointer' }}
                                onClick={() => setSelectedRecord({ disruption_id: id })}
                                title="Click to view historical incident record"
                              >
                                {id}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '24px' }}>
                <ShieldAlert size={22} style={{ color: 'var(--text-muted)' }} />
                <b>No Supplier Segments Matched the Active Filter Criteria</b>
                <span>Try broadening regions, industry verticals, or lowering the minimum disruption severity.</span>
              </div>
            )}
          </Panel>

          {selectedRecord && (
            <HistoricalRecordModal item={selectedRecord} onClose={() => setSelectedRecord(null)} />
          )}
        </>
      )}
    </>
  )
}


function Evaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const r = await api.get('/evaluation');
      setData(r.data);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load() }, []);

  const report = data?.latest_report;
  const evals = data?.evaluations || [];
  const avgScore = report?.average_score_pct ?? (evals.length ? Math.round(evals.reduce((acc, e) => acc + (e.data?.overall_score_pct || e.data?.ragas?.overall_pct || 0), 0) / evals.length) : null);

  return (
    <>
      <PageHeading
        eyebrow="RAGAS QUALITY ASSESSMENT"
        title="RAGAS Evaluation"
        description="Automated evaluation of recommendation faithfulness, answer relevancy, context precision, and context recall via RAGAS."
      />
      {error && <ErrorState title="Evaluation results are unavailable" description={error} onRetry={load} />}
      {loading ? (
        <LoadingSkeleton rows={4} label="Loading RAGAS evaluations" />
      ) : !error && (
        <>
          <div className="stats-grid">
            <Stat label="Evaluation Method" value="RAGAS" note="Single unified framework" tone="blue" />
            <Stat label="Average RAGAS Score" value={avgScore != null ? avgScore : 'N/A'} suffix={avgScore != null ? '%' : ''} note="Evaluated answer & context" tone="green" />
            <Stat label="Target Threshold" value="75.0%" note="Quality baseline mark" tone="amber" />
            <Stat label="Total Recorded Runs" value={evals.length || (report?.total_tests ?? 0)} note="Persisted evaluation records" />
          </div>

          {report?.results && report.results.length > 0 && (
            <Panel title="RAGAS Evaluation Scenarios" subtitle="Evaluated strictly via RAGAS framework (Faithfulness, Relevancy, Precision, Recall)">
              <div className="eval-table-container">
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>Scenario ID</th>
                      <th>Scenario</th>
                      <th>Disruption Type</th>
                      <th>Severity</th>
                      <th>RAGAS Score</th>
                      <th>Status</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.results.map(r => (
                      <tr key={r.test_id}>
                        <td><b>{r.test_id}</b></td>
                        <td>{r.name}</td>
                        <td>{r.disruption_type}</td>
                        <td>Sev {r.severity}/5</td>
                        <td><b>{r.overall_score_pct}%</b></td>
                        <td><span className={`eval-status-badge ${r.status?.toLowerCase().replace(/\s+/g, '-')}`}>{r.status}</span></td>
                        <td>
                          <span className={r.passed ? 'eval-badge-pass' : 'eval-badge-fail'}>
                            {r.passed ? 'PASSED' : 'FAILED'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          <div className="overview-grid">
            <Panel title="Historical RAGAS Evaluation Runs" subtitle="Real evaluation records stored from completed analyses">
              {evals.length > 0 ? (
                <div className="eval-table-container">
                  <table className="eval-table">
                    <thead>
                      <tr>
                        <th>Type</th>
                        <th>Score</th>
                        <th>Status</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evals.map((e, idx) => {
                        const score = e.data?.overall_score_pct ?? e.data?.ragas?.overall_pct;
                        const st = e.data?.status || (score >= 75 ? 'GOOD' : 'NEEDS IMPROVEMENT');
                        return (
                          <tr key={idx}>
                            <td><b>RAGAS</b></td>
                            <td>{score != null ? `${score}%` : 'N/A'}</td>
                            <td><span className={`eval-status-badge ${st.toLowerCase().replace(/\s+/g, '-')}`}>{st}</span></td>
                            <td><small>{e.created_at ? new Date(e.created_at).toLocaleString() : 'Recent'}</small></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <pre className="json-panel">{JSON.stringify(evals.length ? evals : 'No saved RAGAS evaluation records yet. Run a disruption analysis to record results.', null, 2)}</pre>
              )}
            </Panel>
            <Panel title="Saved Model Training Metrics" subtitle="Persisted regression and classification metrics">
              <pre className="json-panel">{JSON.stringify(data?.model_metrics || 'Metrics unavailable', null, 2)}</pre>
            </Panel>
          </div>
        </>
      )}
    </>
  );
}

export default function App() {
  const location=useLocation();const navigate=useNavigate();const [menu,setMenu]=useState(false);const [latest,setLatest]=useState(null);const [health,setHealth]=useState(null);const [toast,setToast]=useState('');const [user,setUser]=useState(null)
  useEffect(()=>{api.get('/health').then(({data})=>setHealth(data)).catch(err=>{console.error('Health check failed',err);setHealth({status:'unavailable'})})},[])
  useEffect(()=>{api.get('/auth/me').then(({data})=>setUser(data)).catch(()=>{})},[])  
  useEffect(()=>{setMenu(false)},[location.pathname])
  useEffect(()=>{if(!toast)return;const timeout=setTimeout(()=>setToast(''),3500);return()=>clearTimeout(timeout)},[toast])
  const logout=async()=>{try{await api.post('/auth/logout')}catch{}sessionStorage.removeItem('chainwise_access_token');navigate('/',{replace:true})}
  const title=location.pathname.endsWith('/analysis')?'Disruption analysis':location.pathname.endsWith('/history')?'History':location.pathname.endsWith('/similar-cases')?'Similar cases':location.pathname.endsWith('/predictions')?'Predictions':location.pathname.endsWith('/recommendations')?'Recommendations':location.pathname.endsWith('/supplier-risk')?'Supplier risk':location.pathname.endsWith('/resilience')?'Risk & resilience':location.pathname.endsWith('/analytics')?'Analytics':location.pathname.endsWith('/feedback')?'Feedback':location.pathname.endsWith('/evaluation')?'Evaluation':'Overview'
  const completed=result=>{setLatest(result);navigate('/dashboard/analysis')}
  return <div className="app-shell"><Sidebar open={menu} close={()=>setMenu(false)} onLogout={logout} user={user}/><div className="main-column"><Header title={title} openMenu={()=>setMenu(true)} health={health}/><main className="page-content"><div className="page-view" key={location.pathname}><Routes><Route index element={<Overview onAnalyze={()=>navigate('/dashboard/analysis')}/>} /><Route path="analysis" element={<AnalysisPage latest={latest} onComplete={completed} onToast={setToast}/>} /><Route path="history" element={<HistoryPage onToast={setToast}/>} /><Route path="similar-cases" element={<SimilarCases/>}/><Route path="predictions" element={<PredictionPage/>}/><Route path="recommendations" element={<Recommendations latest={latest}/>}/><Route path="supplier-risk" element={<SupplierRiskDashboard/>}/><Route path="resilience" element={<Resilience latest={latest}/>}/><Route path="analytics" element={<Analytics/>}/><Route path="feedback" element={<FeedbackPage latest={latest} onToast={setToast}/>}/><Route path="evaluation" element={<Evaluation/>}/></Routes></div><footer className="app-footer"><span><Boxes size={13}/> Chainwise operations center</span><span>Recommendations cite retrieved dataset records.</span></footer></main></div><Toast message={toast} onClose={()=>setToast('')}/></div>
}
