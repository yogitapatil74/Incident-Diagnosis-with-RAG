# Post-Mortem: Payment Service Outage — INC-2024-001
**Date:** 2024-01-15
**Severity:** P1
**Duration:** 75 minutes (14:32 – 15:47 UTC)
**Author:** On-call SRE
**Status:** Resolved

---

## Executive Summary
A deployment of payments-service v2.3.1 introduced a slow database query that
exhausted the HikariCP connection pool within 22 minutes. Approximately 12% of
payment requests failed with HTTP 503 errors during the incident window.
Total estimated revenue impact: ~$14,000.

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 14:10 | payments-service v2.3.1 deployed to production |
| 14:32 | PagerDuty alert: payments-service 5xx error rate > 5% |
| 14:35 | On-call SRE acknowledges alert, begins triage |
| 14:41 | Pod logs show HikariPool timeout errors |
| 14:48 | Database query analysis identifies missing index |
| 14:53 | Decision to rollback to v2.3.0 approved |
| 15:01 | Rollback initiated: `kubectl rollout undo` |
| 15:08 | Error rate returns to baseline (<0.1%) |
| 15:47 | Incident resolved, monitoring confirmed stable |

---

## Root Cause
The v2.3.1 release introduced a new analytics query in the payment history
endpoint that performed a full table scan on the `payments` table (47M rows)
due to a missing composite index on `(customer_id, created_at)`. Under load,
each request held a database connection for 8-15 seconds instead of <50ms,
exhausting the HikariCP pool (max_pool_size=10) within minutes.

**Contributing factors:**
- HikariCP pool size (10) was set 18 months ago and not reviewed as traffic grew 3x
- The slow query was not caught in staging (staging has only 50k rows)
- No slow query threshold alert was configured on the database

---

## Impact
- **Affected users:** ~8,400 customers (12% of concurrent sessions)
- **Failed transactions:** 1,247 payment attempts
- **Error rate peak:** 18% at 14:45 UTC
- **P99 latency peak:** 31 seconds (normal: 180ms)

---

## Resolution
1. Rolled back to payments-service v2.3.0
2. Added index: `CREATE INDEX CONCURRENTLY idx_payments_customer_date ON payments(customer_id, created_at)`
3. Increased pool size: `spring.datasource.hikari.maximum-pool-size=25`
4. Added `statement_timeout = '5s'` on application database role

---

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Add slow query check to CI pipeline using pg_stat_statements baseline | Platform Team | 2024-01-29 | Open |
| Increase staging dataset to 10M rows | Data Team | 2024-02-05 | Open |
| Add Prometheus alert: hikaricp_connections_pending > 5 | SRE | 2024-01-22 | Closed |
| Review all services with pool_size < 20 | SRE | 2024-01-22 | Closed |
| Add connection pool metrics to SLO dashboard | SRE | 2024-01-29 | Open |

---

## Lessons Learned
1. **Pool sizing needs regular review** — traffic growth must trigger config reviews.
2. **Staging data volume matters** — full-table scans are invisible on small datasets.
3. **Statement timeouts are essential** — a per-connection timeout of 5s would have
   limited blast radius significantly.
4. **Deploy with canary** — a 5% canary release would have surfaced this within
   minutes without broad customer impact.

---

## Detection & Response Analysis
- **MTTD:** 3 minutes (alert fired quickly — good)
- **MTTI:** 22 minutes (time from alert to identifying root cause — needs improvement)
- **MTTR:** 75 minutes (acceptable for P1, but rollback decision delayed 18 minutes)

---

## Follow-up
This post-mortem will be reviewed in the weekly SRE sync. Action items tracked in Jira
under project SRE-2024-Q1.
