# Running the chatbot on Kubernetes with Prometheus, Grafana and Jenkins

Local setup on one Windows laptop: Docker Desktop's built-in Kubernetes runs the chatbot,
Prometheus scrapes its metrics and Grafana shows them; Jenkins builds and deploys on each change.
The Cloud Run production service is described separately in `terraform/`.

```
Jenkinsfile ─► tests ─► alert-rule tests ─► docker build ─► kubectl apply -k k8s ─► rollout ─► smoke test
                                                                                     (rollback on failure)
namespace ragbot:
  ragbot      Deployment + Service   :8080 app, :9100 /metrics      (probes on /_stcore/health)
  prometheus  Deployment + Service   :9090  scrapes ragbot:9100, evaluates monitoring/alerts.yml
  grafana     Deployment + Service   :3000  datasource + dashboard provisioned from monitoring/grafana/
```

## What is measured

`ragbot/metrics.py` records counts and timings only, never question text (backlog story S6):
questions by reader and outcome (answered, not_found, refused, error), answer latency, guardrail
blocks by reason, PDF uploads. Alerts in `monitoring/alerts.yml`:

| Alert | When | Why |
|---|---|---|
| RagbotDown | metrics unreachable 2 min | pod crashed or stuck |
| RagbotHighErrorRate | over 10% of questions error for 5 min | LLM provider problem (INC-003, INC-005) |
| RagbotSlowAnswers | p90 above 8 s for 10 min | backlog S5 target |
| RagbotPromptInjectionBurst | more than 5 injection attempts in 15 min | someone probing the app |

`monitoring/alerts_test.yml` unit-tests every alert, including the cases that must *not* fire:
`promtool test rules monitoring/alerts_test.yml`.

## One-time setup (Windows)

1. **Docker Desktop**: install, then Settings > Kubernetes > *Enable Kubernetes* > Apply & restart.
   Choose the **kubeadm** cluster type if asked: it uses the images you build with `docker build`
   directly. Check with `kubectl get nodes` (one node, `Ready`).
2. **Build the image** (from the project folder; takes a while the first time):
   ```
   docker build -t rag-pdf-chatbot:local .
   ```
3. **Secrets** (read from your existing `.env`, so the key is never typed or committed):
   ```
   kubectl create namespace ragbot
   kubectl -n ragbot create secret generic ragbot-secrets --from-env-file=.env
   kubectl -n ragbot create secret generic grafana-admin --from-literal=password=CHOOSE_ONE
   ```
4. **Deploy everything**:
   ```
   kubectl apply -k k8s
   kubectl -n ragbot rollout status deployment/ragbot --timeout=300s
   kubectl -n ragbot get pods
   ```
5. **Open the apps** (one Command Prompt each, leave them running):
   ```
   kubectl -n ragbot port-forward svc/ragbot 8501:8080
   kubectl -n ragbot port-forward svc/prometheus 9090:9090
   kubectl -n ragbot port-forward svc/grafana 3000:3000
   ```
   Chatbot: http://localhost:8501 · Prometheus: http://localhost:9090/alerts ·
   Grafana: http://localhost:3000 (user `admin`, the password from step 3).
   Ask the chatbot a few questions, including one that is not in the PDF; the dashboard
   "Ask the Terms: live usage" fills in within a minute.

## Jenkins

1. Download `jenkins.war` (LTS) from jenkins.io and run it from a Command Prompt, so it uses your
   Docker, kubectl and Python: `java -jar jenkins.war --httpPort=8081`.
2. Open http://localhost:8081, paste the initial admin password it prints, choose
   **Install suggested plugins** and create a user.
3. New Item > **Pipeline** > *Pipeline script from SCM* > Git > your repository URL, branch
   `*/main`, script path `Jenkinsfile`. Save, then **Build Now**.
4. A failed rollout or smoke test runs `kubectl rollout undo`, so the previous version keeps
   serving.

## Useful commands

```
kubectl -n ragbot logs deploy/ragbot --tail=50        # app logs
kubectl -n ragbot rollout history deployment/ragbot   # versions
kubectl -n ragbot rollout undo deployment/ragbot      # roll back
kubectl delete namespace ragbot                       # remove everything
```
