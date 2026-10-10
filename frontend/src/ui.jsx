import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { AlertCircle, ArrowUpRight, Check, LoaderCircle, Sparkles, X } from 'lucide-react'

export function LoadingSkeleton({ rows = 3, label = 'Loading content' }) {
  return <div className="skeleton-stack" role="status" aria-label={label}>
    {Array.from({ length: rows }, (_, index) => <div className="skeleton-row" key={index}>
      <span className="skeleton-block" /><span className="skeleton-lines"><i /><i /></span>
    </div>)}
  </div>
}

export function ErrorState({ title = 'This information is unavailable', description, onRetry }) {
  return <section className="error-state" role="alert"><span className="error-state-icon"><AlertCircle size={18} /></span>
    <div><h3>{title}</h3><p>{description}</p></div>
    {onRetry && <button className="outline-btn" onClick={onRetry}>Try again</button>}
  </section>
}

export function AnalysisProgress() {
  return <section className="analysis-progress" role="status" aria-live="polite">
    <span className="progress-orbit"><Sparkles size={18} /></span>
    <div className="progress-copy"><b>Analyzing disruption...</b><span>Waiting for the complete workflow response</span>
      <div className="progress-track" aria-hidden="true"><i /></div>
      <small>Classification, historical retrieval, predictions and recommendation are part of this analysis.</small>
    </div>
    <LoaderCircle className="progress-spinner" size={18} aria-hidden="true" />
  </section>
}

export function RecoveryTimeline({ response, partial, full }) {
  const items = [['Response', response], ['Partial recovery', partial], ['Full recovery', full]]
  return <div className="recovery-timeline" aria-label="Historical recovery timeline">{items.map(([label, value], index) =>
    <div className="timeline-step" key={label}><span className="timeline-node">{index + 1}</span>
      <span className="timeline-copy"><b>{label}</b><small>{value == null ? 'Not available' : `${value} days`}</small></span>
    </div>)}
  </div>
}

export function HistoricalCaseCard({ item, compact = false, onSelect }) {
  return <article className={`historical-card ${compact ? 'compact-evidence-card' : ''}`}>
    <button className="historical-card-toggle" type="button" aria-haspopup="dialog" aria-label={`View details for ${item.disruption_id}`} onClick={onSelect}>
      <span className="case-id">{item.disruption_id}</span>
      <span className="case-summary"><b>{item.disruption_type} <i>·</i> {item.industry}</b><small>Tier {item.supplier_tier} <i>·</i> {item.supplier_region} <i>·</i> Severity {item.disruption_severity}/5</small></span>
      <span className="case-score">{item.hybrid_score == null ? 'Historical case' : `${Math.round(item.hybrid_score * 100)}% match`}</span>
      <ArrowUpRight size={15} className="case-open-icon" aria-hidden="true" />
    </button>
  </article>
}

const recordMoney = value => value == null ? 'Not available' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)
const recordPercent = value => value == null ? 'Not available' : `${new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value)}%`
const recordValue = value => value == null || value === '' ? 'Not available' : String(value)
const yesNo = value => value == null ? 'Not available' : value ? 'Yes' : 'No'

export function HistoricalRecordModal({ item, onClose }) {
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
      if (event.key === 'Tab') {
        const dialog = document.querySelector('.historical-record-dialog')
        const focusable = dialog?.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])')
        if (!focusable?.length) return
        const first = focusable[0]
        const last = focusable[focusable.length - 1]
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
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

  const score = item.hybrid_score == null ? null : `${Math.round(item.hybrid_score * 100)}%`
  const details = [
    ['Supplier tier', item.supplier_tier], ['Supplier region', item.supplier_region],
    ['Supplier size', item.supplier_size], ['Disruption severity', item.disruption_severity == null ? null : `${item.disruption_severity}/5`],
    ['Backup supplier', yesNo(item.has_backup_supplier)], ['Permanent supplier change', yesNo(item.permanent_supplier_change)],
    ['Semantic score', recordPercent(item.semantic_score == null ? null : item.semantic_score * 100)],
    ['Keyword score', recordPercent(item.keyword_score == null ? null : item.keyword_score * 100)],
  ]

  return createPortal(<div className={`historical-record-backdrop ${open ? 'is-open' : ''}`} onMouseDown={event => { if (event.target === event.currentTarget) requestClose() }}>
    <section className="historical-record-dialog" role="dialog" aria-modal="true" aria-labelledby="historical-record-title" aria-describedby="historical-record-description">
      <header className="historical-record-header">
        <div><span className="historical-record-kicker">HISTORICAL EVIDENCE</span><h2 id="historical-record-title">{item.disruption_id}</h2><p id="historical-record-description">{item.disruption_type} <i>·</i> {item.industry}</p></div>
        <button ref={closeButtonRef} className="historical-record-close" type="button" onClick={requestClose} aria-label="Close historical record details"><X size={18}/></button>
      </header>
      <div className="historical-record-content">
        <div className="historical-record-score-row"><span>Match score</span><b>{score || 'Historical record'}</b><small>Retrieval score, not a probability</small></div>
        <div className="historical-record-metrics">
          <article><small>Production impact</small><b>{recordPercent(item.production_impact_pct)}</b></article>
          <article><small>Financial loss</small><b>{recordMoney(item.revenue_loss_usd)}</b></article>
          <article><small>Historical response</small><b>{recordValue(item.response_type)}</b></article>
        </div>
        <section className="historical-record-section"><h3>Recovery timeline</h3><RecoveryTimeline response={item.response_time_days} partial={item.partial_recovery_days} full={item.full_recovery_days}/></section>
        <section className="historical-record-section"><h3>Record details</h3><dl className="historical-record-details">{details.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{recordValue(value)}</dd></div>)}</dl></section>
      </div>
    </section>
  </div>, document.body)
}

export function Toast({ message, onClose }) {
  if (!message) return null
  return <div className="toast" role="status"><span><Check size={16} /></span>{message}<button onClick={onClose} aria-label="Dismiss notification">×</button></div>
}
