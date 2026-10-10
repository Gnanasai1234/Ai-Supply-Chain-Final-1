from app.models import Disruption


def disruption_to_document(row: Disruption | dict) -> str:
    get = row.get if isinstance(row, dict) else lambda key: getattr(row, key)
    yes_no = lambda value: "Yes" if bool(value) else "No"
    return (
        f"Disruption ID: {get('disruption_id')}\n\n"
        f"Disruption type: {get('disruption_type')}. Industry: {get('industry')}. "
        f"Supplier tier: Tier {get('supplier_tier')}. Supplier region: {get('supplier_region')}. "
        f"Supplier size: {get('supplier_size')}. Backup supplier available: {yes_no(get('has_backup_supplier'))}. "
        f"Disruption severity: {get('disruption_severity')}/5.\n\n"
        f"Production impact: {float(get('production_impact_pct')):.1f}%. "
        f"Revenue loss: ${float(get('revenue_loss_usd')):,.2f}.\n\n"
        f"Historical response: {get('response_type')}. Response time: {get('response_time_days')} days. "
        f"Partial recovery: {get('partial_recovery_days')} days. Full recovery: {get('full_recovery_days')} days. "
        f"Permanent supplier change: {yes_no(get('permanent_supplier_change'))}."
    )


def disruption_metadata(row: Disruption | dict) -> dict:
    get = row.get if isinstance(row, dict) else lambda key: getattr(row, key)
    return {
        "disruption_id": str(get("disruption_id")),
        "disruption_type": str(get("disruption_type")),
        "industry": str(get("industry")),
        "supplier_tier": str(get("supplier_tier")),
        "supplier_region": str(get("supplier_region")),
        "supplier_size": str(get("supplier_size")),
        "has_backup_supplier": bool(get("has_backup_supplier")),
        "disruption_severity": int(get("disruption_severity")),
        "response_type": str(get("response_type")),
    }
