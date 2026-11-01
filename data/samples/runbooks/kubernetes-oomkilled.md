# Runbook: Kubernetes OOMKilled / CrashLoopBackOff

## Overview
This runbook covers pods that are killed by the Kubernetes OOM killer
and enter CrashLoopBackOff state. OOMKilled means the container exceeded
its configured memory limit.

## Symptoms
- Pod status shows `OOMKilled` or `CrashLoopBackOff`
- `kubectl describe pod` shows `Reason: OOMKilled` in Last State
- Container restart count incrementing in `kubectl get pods`
- Memory usage metric approaching or exceeding the limit

## Immediate Triage

### 1. Identify affected pods
```bash
kubectl get pods --all-namespaces | grep -E "CrashLoop|OOMKilled"
kubectl describe pod <pod-name> -n <namespace>
```

### 2. Review memory usage history
```bash
# Check current memory usage
kubectl top pods -n <namespace>

# Check limits and requests
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[*].resources}'
```

### 3. Check recent memory trends in Grafana/Prometheus
```promql
container_memory_working_set_bytes{pod=~"<pod-prefix>.*"}
  / container_spec_memory_limit_bytes{pod=~"<pod-prefix>.*"}
```

## Root Cause Analysis

### Common root causes:
1. **Memory limit set too low** — service grew, limits not updated
2. **Memory leak** — unbounded cache growth, unreleased objects
3. **Large in-memory data load** — ML model load, bulk data processing
4. **JVM heap misconfiguration** — Xmx not aligned with container limits
5. **Sudden traffic spike** — per-request memory multiplied by concurrent requests

### For JVM services — check heap vs container alignment:
```bash
# Container limit e.g. 2Gi → JVM Xmx should be ~75% = 1.5Gi
# If Xmx not set, JVM may auto-size to >container limit
kubectl exec <pod> -- java -XX:+PrintFlagsFinal -version 2>&1 | grep -i heapsize
```

## Remediation Steps

### Immediate
1. Increase memory limits in the deployment:
```bash
kubectl set resources deployment/<name> \
  --limits=memory=4Gi \
  --requests=memory=2Gi \
  -n <namespace>
```

2. For JVM services, align heap with new limit:
```bash
kubectl set env deployment/<name> \
  JAVA_OPTS="-Xms1g -Xmx3g -XX:+UseContainerSupport" \
  -n <namespace>
```

3. If cause is a scheduled job (batch/ML model load), check cron schedule:
```bash
kubectl get cronjobs -n <namespace>
kubectl logs job/<job-name> -n <namespace>
```

### Short-term
4. Add memory-based Horizontal Pod Autoscaler:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

5. Add heap dump on OOM for investigation:
```
-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/heapdump.hprof
```

### Long-term
6. Implement memory profiling in staging before production deploys.
7. Add Prometheus alert: memory utilisation > 85% for 5 minutes.
8. Set LimitRange in namespace to enforce minimum request/limit ratios.

## Escalation
Escalate to platform/SRE team if:
- OOM occurs even after limit increase (potential leak)
- Heap dumps show unexpected object retention
- Issue affects multiple services simultaneously (node-level memory pressure)

## Related Incidents
- INC-2024-002: Recommendation engine OOMKilled after ML model update
- INC-2023-145: Order service JVM heap misconfiguration post-migration

## Post-Resolution Checklist
- [ ] Memory limits updated in Helm values and GitOps repo
- [ ] Alert threshold configured in alerting rules
- [ ] Load test run with new limits in staging
- [ ] Memory profiling added to deployment checklist
