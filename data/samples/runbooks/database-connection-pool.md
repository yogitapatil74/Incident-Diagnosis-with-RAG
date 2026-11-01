# Runbook: Database Connection Pool Exhaustion

## Overview
This runbook covers diagnosis and remediation of database connection pool exhaustion
in cloud-native services using HikariCP, pgBouncer, or similar pooling layers.

## Symptoms
- HTTP 503 or 500 errors from services connecting to PostgreSQL/MySQL
- Log messages: `Connection is not available, request timed out`
- Elevated pod restart counts in Kubernetes
- Database active connections near or at `max_connections` limit
- Increased request latency before failures begin

## Immediate Triage (first 5 minutes)

### 1. Check active database connections
```sql
SELECT count(*), state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE datname = '<your_database>'
GROUP BY state, wait_event_type, wait_event
ORDER BY count DESC;
```

### 2. Identify blocking queries
```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND now() - pg_stat_activity.query_start > interval '30 seconds'
ORDER BY duration DESC;
```

### 3. Check HikariCP pool metrics (if exposed via JMX/Prometheus)
```
hikaricp_connections_active   # currently in-use connections
hikaricp_connections_idle     # idle connections available
hikaricp_connections_pending  # threads waiting for a connection
hikaricp_connections_timeout_total  # total timeout events
```

## Root Cause Analysis

### Common root causes (ordered by frequency):
1. **Slow query regression** — a new query introduced by deployment takes 10-100x longer,
   holding connections longer and exhausting the pool.
2. **Pool size too small** — traffic growth outpaced pool configuration.
3. **Connection leak** — connections are not properly returned to pool (missing try-with-resources).
4. **Long-running transactions** — batch jobs or analytics queries holding connections.
5. **Database-side overload** — underlying DB CPU/IO saturated, slowing all queries.

## Remediation Steps

### Immediate (stop the bleeding)
1. Identify and kill the longest-running queries:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE duration > interval '2 minutes'
     AND state = 'idle in transaction';
   ```
2. If a recent deployment caused regression, initiate rollback:
   ```bash
   kubectl rollout undo deployment/<service-name>
   kubectl rollout status deployment/<service-name>
   ```
3. Temporarily increase pool size as emergency relief:
   ```bash
   kubectl set env deployment/<service> HIKARI_MAXIMUM_POOL_SIZE=50
   ```

### Short-term
4. Add missing indexes on slow queries using EXPLAIN ANALYZE.
5. Set statement_timeout on application connections to prevent runaway queries:
   ```
   spring.datasource.hikari.connection-init-sql=SET statement_timeout TO '5s'
   ```

### Long-term
6. Implement pgBouncer in transaction pooling mode to reduce per-service connection pressure.
7. Add pre-deployment slow query CI check using pgBadger or pg_stat_statements baseline.
8. Set up Prometheus alerting on `hikaricp_connections_pending > 5` for 2+ minutes.

## Escalation
Escalate to database team (DBA on-call) if:
- Database `max_connections` is reached and increasing it requires restart
- Root cause is storage-level I/O saturation (check `pg_stat_bgwriter`)
- Issue persists after pool size increase and query kill

## Related Incidents
- INC-2024-001: Payment service connection pool exhaustion (slow query, v2.3.1)
- INC-2023-089: Checkout service connection leak (missing finally block)

## Post-Resolution Checklist
- [ ] Slow query added to performance regression test suite
- [ ] Pool size documented in service runbook
- [ ] Alert on `connections_pending` configured
- [ ] Post-mortem scheduled within 48 hours
