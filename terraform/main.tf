# The Cloud Run deployment that was first set up by hand (DEPLOYMENT.md, INC-005), described as
# code. imports.tf adopts the existing resources, so the first apply changes nothing in the cloud.

data "google_project" "this" {}

locals {
  # The default compute service account runs the service and builds it with Cloud Build.
  runtime_sa = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"
}

# Secret containers only. The secret VALUES stay in Secret Manager and never enter Terraform state.
resource "google_secret_manager_secret" "groq_api_key" {
  secret_id = "groq-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "app_password" {
  secret_id = "app-password"
  replication {
    auto {}
  }
}

# Roles granted to the runtime service account during the first deploy (INC-005).
resource "google_project_iam_member" "runtime_reads_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = local.runtime_sa
}

resource "google_project_iam_member" "runtime_builds" {
  project = var.project_id
  role    = "roles/run.builder"
  member  = local.runtime_sa
}

resource "google_cloud_run_v2_service" "ragbot" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    session_affinity = true

    scaling {
      min_instance_count = 0
      max_instance_count = 2 # caps cost
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
      }

      env {
        name  = "LLM_BASE_URL"
        value = "https://api.groq.com/openai/v1"
      }
      env {
        name  = "LLM_MODEL"
        value = "openai/gpt-oss-20b"
      }
      env {
        name  = "DEFAULT_READER"
        value = "llm"
      }
      env {
        name = "LLM_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.groq_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "APP_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.app_password.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  lifecycle {
    # Each deploy builds a new image and stamps the revision; that is the pipeline's job.
    ignore_changes = [
      template[0].containers[0].image,
      template[0].revision,
      template[0].labels,
      template[0].annotations,
      client,
      client_version,
    ]
  }
}

# The app is public behind its own password gate (web_guard.py).
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.ragbot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.ragbot.uri
}
