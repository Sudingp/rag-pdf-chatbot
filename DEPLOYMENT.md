# Deployment runbook

How the live app is built, deployed, checked and rolled back. Incidents from past deploys are in
[INCIDENTS.md](INCIDENTS.md).

| | |
|---|---|
| **Service** | `rag-pdf-chatbot` on Google Cloud Run, region `asia-south1` (Mumbai) |
| **URL** | https://rag-pdf-chatbot-631970366044.asia-south1.run.app (password-protected) |
| **Image** | Built from `Dockerfile` by Cloud Build, stored in Artifact Registry (`cloud-run-source-deploy/rag-pdf-chatbot`) |
| **Secrets** | `groq-api-key` and `app-password` in Secret Manager, mounted as `LLM_API_KEY` and `APP_PASSWORD` |
| **Scaling** | 0 to 2 instances, 2 GiB memory, 1 CPU, session affinity |
| **Pipeline** | `azure-pipelines.yml` on Azure DevOps: Test, then Security, then Deploy (manual approval) |

## 1. Normal deploy (through the pipeline)

1. Push to `main` (or merge a pull request into it).
2. Azure Pipelines runs **Test** (pytest, results published to the run's Tests tab) and
   **Security** (pip-audit on the app's own dependencies). Either failing stops the run.
3. **Deploy** waits for approval on the `production` environment. Approve it in Azure DevOps.
4. The deploy step runs `gcloud run deploy --source .`: Cloud Build builds the image and Cloud Run
   rolls out a new revision. Settings not passed on the command line (env vars, secrets,
   memory, scaling) are kept from the current revision.
5. The smoke test calls `/_stcore/health` and fails the run unless it returns `ok`.
6. Open the URL, log in, upload a PDF and ask one question as a final manual check.

## 2. Manual deploy (if the pipeline is unavailable)

Windows Command Prompt, from the project folder:

```
gcloud run deploy rag-pdf-chatbot --source . --region asia-south1 --quiet
```

If gcloud crashes with a network error while Cloud Build is still running (INC-005), do not start
again. Wait for the build to show **Success** in the Cloud Build console, then deploy that image:

```
gcloud run deploy rag-pdf-chatbot --image asia-south1-docker.pkg.dev/project-e73aa35c-f435-49fa-b3e/cloud-run-source-deploy/rag-pdf-chatbot --region asia-south1
```

## 3. Check what is running

```
gcloud run services describe rag-pdf-chatbot --region asia-south1 --format "value(status.latestReadyRevisionName,status.url)"
gcloud run revisions list --service rag-pdf-chatbot --region asia-south1
```

## 4. Read the logs

Reader failures are logged by `app.py` (`rag.ask failed: ...`) and never shown to visitors.

```
gcloud run services logs read rag-pdf-chatbot --region asia-south1 --limit 50
```

## 5. Roll back

Send all traffic back to the last good revision (names come from the revisions list above):

```
gcloud run services update-traffic rag-pdf-chatbot --region asia-south1 --to-revisions REVISION_NAME=100
```

The bad revision stays available for debugging but serves no traffic. A later deploy creates a
new revision; check afterwards that it receives 100% of the traffic again
(`gcloud run services update-traffic rag-pdf-chatbot --region asia-south1 --to-latest`).

## 6. Rotate a secret

1. Add a new version in Secret Manager (Console: Security > Secret Manager > `groq-api-key` >
   **New version**, paste the new key there, never into a command or a file in the repo).
2. Secrets are mounted as `:latest`, which is read when an instance starts, so roll out a new
   revision to pick it up: run the manual deploy in section 2, or re-run the pipeline.
3. Disable the old version once the app answers a question with the new key.

## 7. Pipeline agent

The pipeline uses one self-hosted Windows agent (free tier: one self-hosted parallel job).

- Installed in `C:\agents\rag`, registered to the `Default` pool with a personal access token
  scoped to **Agent Pools (read, manage)** only.
- Started with `run.cmd` from a normal Command Prompt, so it uses the logged-in gcloud account.
  Running it as a Windows service would run as a different user without those credentials.
- Needs Python, Git and the Google Cloud CLI on the PATH. The agent only runs while the laptop is
  on and `run.cmd` is open; queued runs start when it comes back.

## 8. Switch the live app to Gemini on Vertex AI

No key is created or stored: Cloud Run's runtime service account is allowed to call Vertex AI.

```
gcloud services enable aiplatform.googleapis.com
gcloud run services describe rag-pdf-chatbot --region asia-south1 --format="value(spec.template.spec.serviceAccountName)"
```

If that prints nothing, the service uses the default compute account,
`631970366044-compute@developer.gserviceaccount.com`. Grant it the role, then set the reader:

```
gcloud projects add-iam-policy-binding project-e73aa35c-f435-49fa-b3e --member="serviceAccount:<account from above>" --role="roles/aiplatform.user"
gcloud run services update rag-pdf-chatbot --region asia-south1 --update-env-vars VERTEX_PROJECT=project-e73aa35c-f435-49fa-b3e,DEFAULT_READER=vertex
```

The image must include `google-auth` (it is in `requirements-app.txt`), so deploy once after
pulling this change. Check with a question in the app; a 403 in the logs means the role is missing.
