# Cloud Run as code (Terraform)

The production service was first created by hand with `gcloud` (DEPLOYMENT.md). This folder
describes the same service, its secrets and its IAM roles as code, and **imports** the existing
resources instead of creating new ones.

| Resource | What it is |
|---|---|
| `google_cloud_run_v2_service.ragbot` | the service: 1 CPU, 2 GiB, 0 to 2 instances, session affinity, env vars, secrets |
| `google_secret_manager_secret.*` | the two secret containers (values stay in Secret Manager, never in state) |
| `google_project_iam_member.*` | the roles the runtime account needed (INC-005) |
| `google_cloud_run_v2_service_iam_member.public` | public access; the app has its own password gate |

The container image, revision name and deploy annotations are ignored, because each deploy
changes them; Terraform owns the configuration, the pipeline owns the image.

## Use it (Windows)

```
winget install Hashicorp.Terraform
gcloud auth application-default login
cd terraform
copy terraform.tfvars.example terraform.tfvars      (then edit project_id and image)
terraform init
terraform plan
```

The goal of the first plan is **"6 to import, 0 to add, 0 to change, 0 to destroy"**. If it
shows changes, the code and the live service disagree on a setting; adjust the code (not the
service) until the plan is clean, then:

```
terraform apply
```

This only records the six resources in `terraform.tfstate`; nothing in the cloud changes. From
then on, a setting is changed by editing the code and running `plan` and `apply`. The state file
is git-ignored; keep it, or move it to a GCS bucket backend before more than one person uses it.
