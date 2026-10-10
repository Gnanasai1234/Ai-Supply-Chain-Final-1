import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { CalendarDays, ChevronLeft, ChevronRight, Clock3, Eye, Search, Sparkles, Trash2, X } from 'lucide-react'
import api from './api'
import { ErrorState, LoadingSkeleton } from './ui'

const PAGE_SIZE = 10
const errorMessage = error => {
  const detail = error.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(item => item.msg).filter(Boolean).join(' ')
  if (!error.response) return 'The API could not be reached. Check that the backend is running and try again.'
  return 'The request could not be completed. Please try again.'
}
const dateLabel = value => value ? new Date(value).toLocaleString() : 'Date unavailable'
const providerLabel = item => item.final_provider || 'Provider unavailable'
const fmt = value => value == null ? 'Not available' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value)
const money = value => value == null ? 'Not available' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)

function PriorityBadge({ level }) {
  const normalized = (level || 'unavailable').toLowerCase()
  return <span className={`risk-badge ${normalized}`}><i />{level || 'Priority unavailable'}</span>
}

function HistoryStat({ label, value, note }) {
  return <article className="history-stat panel"><span>{label}</span><b>{value}</b>{note && <small>{note}</small>}</article>
}

function AnalysisDetailsModal({ item, onClose }) {
  const [open, setOpen] = useState(false)
  const closingRef = useRef(false)
  const requestCloseRef = useRef(null)
  const closeButtonRef = useRef(null)
  const closeTimerRef = useRef(null)

  useEffect(() => {
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    const animationFrame = window.requestAnimationFrame(() => setOpen(true))
    document.body.style.overflow = 'hidden'
    closeButtonRef.current?.focus()
    const handleKeyDown = event => {
      if (event.key === 'Escape') {
        event.preventDefault()
        requestCloseRef.current?.()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      window.cancelAnimationFrame(animationFrame)
      window.clearTimeout(closeTimerRef.current)
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [])

  const requestClose = () => {
    if (closingRef.current) return
    closingRef.current = true
    setOpen(false)
    closeTimerRef.current = window.setTimeout(onClose, 240)
  }
  requestCloseRef.current = requestClose

  const response = item.response || {}
  const classification = response.classification || {}
  const priority = response.priority || {}
  const preds = response.predictions?.predictions || {}
  const rec = response.recommendation || {}
  const attempts = item.provider_attempts || []
  const similarCases = response.similar_cases || []
  const resilience = response.resilience_analysis || {}

  return createPortal(
    <div
      className={`history-modal-backdrop ${open ? 'is-open' : ''}`}
      onMouseDown={event => {
        if (event.target === event.currentTarget) requestClose()
      }}
      role="presentation"
    >
      <section
        className="history-modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="history-modal-title"
        aria-describedby="history-modal-description"
      >
        <header className="history-modal-header">
          <div>
            <span className="history-modal-kicker">ANALYSIS DETAILS</span>
            <h2 id="history-modal-title">
              {item.disruption_type || classification.disruption_type || 'Disruption Analysis'} · {item.industry || classification.industry || 'Incident'}
            </h2>
            <p id="history-modal-description">{item.query}</p>
          </div>
          <button
            ref={closeButtonRef}
            className="history-modal-close"
            type="button"
            onClick={requestClose}
            aria-label="Close analysis details modal"
          >
            <X size={18} />
          </button>
        </header>

        <div className="history-modal-content">
          {/* Priority & Metadata Banner */}
          <div className="history-modal-banner">
            <div className="history-modal-banner-top">
              <div className="history-modal-chips">
                <span className="history-chip severity">Severity {item.severity ?? classification.severity ?? '?'}/5</span>
                <span className="history-chip">{item.industry || classification.industry || 'Industry'}</span>
                <span className="history-chip">{item.supplier_region || classification.region || 'Region'}</span>
                <span className="history-chip">Tier {item.supplier_tier || classification.tier || '2'}</span>
              </div>
              <div className="history-modal-score-wrap">
                <span>Priority score</span>
                <b>{priority.priority_score == null ? 'Not available' : `${priority.priority_score}/100`}</b>
                <PriorityBadge level={priority.priority_level} />
              </div>
            </div>
            {priority.priority_score != null && (
              <div className="history-modal-progress">
                <em style={{ width: `${Math.max(0, Math.min(100, priority.priority_score))}%` }} />
              </div>
            )}
          </div>

          {/* Predictions Grid */}
          <div className="history-modal-section">
            <h3>Model Predictions</h3>
            <div className="history-modal-metrics">
              <article>
                <small>Production Impact</small>
                <b>{preds.production_impact?.prediction == null ? 'Not available' : `${fmt(preds.production_impact.prediction)}%`}</b>
              </article>
              <article>
                <small>Financial Loss</small>
                <b>{preds.financial_loss?.prediction == null ? 'Not available' : money(preds.financial_loss.prediction)}</b>
              </article>
              <article>
                <small>Predicted Recovery</small>
                <b>{preds.recovery_time?.prediction == null ? 'Not available' : `${fmt(preds.recovery_time.prediction)} days`}</b>
              </article>
              <article>
                <small>Permanent Supplier Change</small>
                <b>{preds.permanent_supplier_change?.available === false ? 'Model not trained' : preds.permanent_supplier_change?.prediction == null ? 'Not available' : `${preds.permanent_supplier_change.prediction ? 'Yes' : 'No'}${preds.permanent_supplier_change.probability ? ` · score ${fmt(preds.permanent_supplier_change.probability * 100)}%` : ''}`}</b>
              </article>
            </div>
          </div>

          {/* Recommendation & Response Summary */}
          <div className="history-modal-section">
            <h3>Recommendation & Strategy</h3>
            <div className="history-modal-rec-card">
              <div className="history-modal-rec-top">
                <Sparkles size={16} />
                <b>{rec.recommended_response || 'Not available'}</b>
                {rec.confidence != null && <small>Evidence share: {fmt(rec.confidence * 100)}%</small>}
              </div>
              <p>{rec.summary || item.summary || 'No summary available.'}</p>
              {rec.reasoning?.length > 0 && (
                <ul className="history-modal-reasoning">
                  {rec.reasoning.map((reason, i) => (
                    <li key={i}>{reason}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Analysis & System Metadata Grid */}
          <div className="history-modal-section">
            <h3>Execution & System Details</h3>
            <div className="history-modal-meta-grid">
              <div><small>Provider</small><b>{providerLabel(item)}</b></div>
              <div><small>Model</small><b>{item.final_model || 'Unavailable'}</b></div>
              <div><small>Fallback Used</small><b>{item.fallback_used ? 'Yes' : 'No'}</b></div>
              <div><small>Response Time</small><b>{item.response_time_ms == null ? 'Unavailable' : `${(item.response_time_ms / 1000).toFixed(2)}s`}</b></div>
              <div><small>Created</small><b>{dateLabel(item.created_at)}</b></div>
              <div><small>Retrieved Cases</small><b>{similarCases.length}</b></div>
              <div><small>Resilience Score</small><b>{resilience.resilience_score != null ? `${resilience.resilience_score}/100` : '—'}</b></div>
              <div><small>Risk Level</small><b>{resilience.risk_level || '—'}</b></div>
            </div>
          </div>

          {/* Provider attempts if present */}
          {attempts.length > 0 && (
            <div className="history-modal-section">
              <h3>Provider Attempts</h3>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr><th>Provider</th><th>Model</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {attempts.map((attempt, index) => (
                      <tr key={`${attempt.provider}-${index}`}>
                        <td>{attempt.provider}</td>
                        <td>{attempt.model}</td>
                        <td><span className={`status-pill ${attempt.status === 'success' ? 'status-resolved' : 'status-active'}`}>{attempt.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Raw JSON */}
          <details className="history-json">
            <summary>Full analysis response (JSON)</summary>
            <pre className="json-panel">{JSON.stringify(response, null, 2)}</pre>
          </details>
        </div>
      </section>
    </div>,
    document.body
  )
}

export default function HistoryPage({ onToast }) {
  const [stats, setStats] = useState(null)
  const [facets, setFacets] = useState({})
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  const [filters, setFilters] = useState({ date: 'All time', provider: 'All', model: 'All', severity: 'All', disruption_type: 'All', industry: 'All', supplier_region: 'All', search: '', date_from: '', date_to: '' })
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [detail, setDetail] = useState(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [filterError, setFilterError] = useState('')

  useEffect(() => {
    let active = true
    Promise.all([api.get('/history/stats'), api.get('/history/filters')]).then(([statsResponse, facetsResponse]) => {
      if (!active) return
      setStats(statsResponse.data)
      setFacets(facetsResponse.data)
    }).catch(err => { if (active) setError(errorMessage(err)) })
    return () => { active = false }
  }, [refresh])

  const params = useMemo(() => {
    const result = { page, page_size: PAGE_SIZE }
    const datePresets = { Today: 'today', 'Last 7 days': 'last_7_days', 'Last 30 days': 'last_30_days', 'Custom range': 'custom' }
    if (datePresets[filters.date]) result.date_preset = datePresets[filters.date]
    if (filters.date === 'Custom range') {
      if (filters.date_from) result.date_from = filters.date_from
      if (filters.date_to) result.date_to = filters.date_to
    }
    for (const key of ['provider', 'model', 'disruption_type', 'industry', 'supplier_region']) {
      if (filters[key] !== 'All') result[key] = filters[key]
    }
    if (filters.severity !== 'All') result.severity = Number(filters.severity)
    if (filters.search.trim()) result.search = filters.search.trim()
    return result
  }, [filters, page])

  useEffect(() => {
    if (filters.date === 'Custom range' && (!filters.date_from || !filters.date_to || filters.date_from > filters.date_to)) {
      setFilterError('Choose a valid start and end date to filter your history.')
      setItems([])
      setTotal(0)
      setTotalPages(0)
      setLoading(false)
      return
    }
    setFilterError('')
    const controller = new AbortController()
    setLoading(true)
    setError('')
    api.get('/history', { params, signal: controller.signal }).then(({ data }) => {
      setItems(data.items || [])
      setTotal(data.total || 0)
      setTotalPages(data.total_pages || 0)
    }).catch(err => {
      if (err.code !== 'ERR_CANCELED') setError(errorMessage(err))
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [params, refresh, filters.date, filters.date_from, filters.date_to])

  const updateFilter = (key, value) => {
    setFilters(current => ({ ...current, [key]: value }))
    setPage(1)
    setDetail(null)
  }
  const clearFilters = () => {
    setFilters({ date: 'All time', provider: 'All', model: 'All', severity: 'All', disruption_type: 'All', industry: 'All', supplier_region: 'All', search: '', date_from: '', date_to: '' })
    setPage(1)
    setDetail(null)
  }
  const openDetail = async id => {
    setError('')
    try { const { data } = await api.get(`/history/${id}`); setDetail(data) }
    catch (err) { setError(errorMessage(err)) }
  }
  const deleteItem = async id => {
    if (!window.confirm('Delete this analysis from your history?')) return
    setError('')
    try {
      await api.delete(`/history/${id}`)
      if (detail?.id === id) setDetail(null)
      setRefresh(value => value + 1)
      onToast?.('Analysis deleted from your history')
    } catch (err) { setError(errorMessage(err)) }
  }
  const clearHistory = async () => {
    setError('')
    try {
      const { data } = await api.delete('/history')
      setConfirmClear(false)
      setDetail(null)
      setPage(1)
      setRefresh(value => value + 1)
      onToast?.(`Deleted ${data.deleted} history item${data.deleted === 1 ? '' : 's'}`)
    } catch (err) { setError(errorMessage(err)) }
  }
  const providerCounts = stats?.provider_counts || {}
  const selectFilter = (key, label, values = []) => <label key={key}>{label}<select value={filters[key]} onChange={event => updateFilter(key, event.target.value)}><option>All</option>{values.map(value => <option key={value}>{value}</option>)}</select></label>

  return <>
    <div className="page-heading"><div><div className="eyebrow">PRIVATE ACTIVITY</div><h1>Analysis history</h1><p>Review, filter and manage analyses saved to your account.</p></div></div>
    {error && <ErrorState title="History could not be loaded" description={error} onRetry={() => setRefresh(value => value + 1)} />}
    {stats && <div className="history-stats-grid">
      <HistoryStat label="Total analyses" value={stats.total_analyses} />
      <HistoryStat label="Successful requests" value={stats.successful_requests} />
      <HistoryStat label="Fallback requests" value={stats.fallback_requests} />
      <HistoryStat label="Average response" value={stats.average_response_time_ms == null ? '—' : `${(stats.average_response_time_ms / 1000).toFixed(2)}s`} />
      {Object.entries(providerCounts).map(([provider, count]) => <HistoryStat key={provider} label={provider} value={count} note="Final response provider" />)}
    </div>}

    <section className="panel history-filters-panel">
      <div className="panel-header"><div><h2>Filter history</h2><p>Filters and provider options apply only to your account.</p></div><button type="button" className="outline-btn" onClick={clearFilters}>Reset</button></div>
      <div className="history-filter-grid">
        <label>Date<select value={filters.date} onChange={event => updateFilter('date', event.target.value)}>{['All time', 'Today', 'Last 7 days', 'Last 30 days', 'Custom range'].map(option => <option key={option}>{option}</option>)}</select></label>
        {selectFilter('provider', 'Provider', facets.providers)}
        {selectFilter('model', 'Model', facets.models)}
        <label>Severity<select value={filters.severity} onChange={event => updateFilter('severity', event.target.value)}><option>All</option>{[1, 2, 3, 4, 5].map(value => <option key={value}>{value}</option>)}</select></label>
        {selectFilter('disruption_type', 'Disruption type', facets.disruption_types)}
        {selectFilter('industry', 'Industry', facets.industries)}
        {selectFilter('supplier_region', 'Supplier region', facets.supplier_regions)}
        <label className="history-search"><span>Search</span><span className="history-search-input"><Search size={15}/><input value={filters.search} onChange={event => updateFilter('search', event.target.value)} placeholder="Query, response, type…" /></span></label>
        {filters.date === 'Custom range' && <>
          <label>From<input type="date" value={filters.date_from} onChange={event => updateFilter('date_from', event.target.value)} /></label>
          <label>To<input type="date" value={filters.date_to} onChange={event => updateFilter('date_to', event.target.value)} /></label>
        </>}
      </div>
      {filterError && <p className="history-filter-error" role="alert">{filterError}</p>}
    </section>

    <section className="panel history-list-panel">
      <div className="panel-header"><div><h2>Your analyses</h2><p>{total} matching analysis{total === 1 ? '' : 'es'}</p></div><button className="outline-btn danger-outline" disabled={total === 0} onClick={() => setConfirmClear(true)}><Trash2 size={14}/> Clear my history</button></div>
      {confirmClear && <div className="history-confirm" role="alertdialog" aria-label="Confirm clear history"><span>This permanently removes all analyses from your account history.</span><div><button className="outline-btn" onClick={() => setConfirmClear(false)}>Cancel</button><button className="primary-btn danger-button" onClick={clearHistory}>Confirm clear history</button></div></div>}
      {loading ? <LoadingSkeleton rows={3} label="Loading your analysis history" /> : filterError ? null : error ? null : items.length ? <div className="history-record-list">{items.map(item => <article className="history-record" key={item.id}>
        <div className="history-record-heading"><div><span className="history-record-date"><CalendarDays size={13}/>{dateLabel(item.created_at)}</span><h3>{item.query}</h3></div><span className="history-provider-chip">{providerLabel(item)}</span></div>
        <p className="history-record-summary">{item.summary || 'Analysis completed.'}</p>
        <div className="history-record-meta"><span><b>Model</b> {item.final_model}</span><span><b>Severity</b> {item.severity}/5</span><span><b>Industry</b> {item.industry}</span><span><b>Region</b> {item.supplier_region}</span><span><b>Response</b> {item.recommended_response || 'Not available'}</span><span><Clock3 size={13}/> {item.response_time_ms == null ? 'Unavailable' : `${(item.response_time_ms / 1000).toFixed(2)}s`}</span><span><b>Fallback</b> {item.fallback_used ? 'Yes' : 'No'}</span></div>
        <div className="history-record-actions"><button className="outline-btn" onClick={() => openDetail(item.id)}><Eye size={14}/> View details</button><button className="outline-btn danger-outline" onClick={() => deleteItem(item.id)}><Trash2 size={14}/> Delete</button></div>
      </article>)}</div> : <div className="empty-state"><CalendarDays size={20}/><b>No matching analyses</b><span>Analyses you run will appear here with provider, model and fallback details.</span></div>}
      {!loading && total > 0 && <div className="history-pagination"><button className="outline-btn" disabled={page <= 1} onClick={() => setPage(value => value - 1)}><ChevronLeft size={15}/> Previous</button><span>Page {page} of {Math.max(1, totalPages)}</span><button className="outline-btn" disabled={page >= totalPages} onClick={() => setPage(value => value + 1)}>Next <ChevronRight size={15}/></button></div>}
    </section>
    {detail && <AnalysisDetailsModal item={detail} onClose={() => setDetail(null)} />}
  </>
}
