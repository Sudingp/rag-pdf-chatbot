# Adopt the resources that already exist instead of creating new ones (Terraform >= 1.6: ids may use variables and data sources).
# After the first successful apply these blocks can stay; they do nothing once imported.

import {
  to = google_secret_manager_secret.groq_api_key
  id = "projects/${var.project_id}/secrets/groq-api-key"
}

import {
  to = google_secret_manager_secret.app_password
  id = "projects/${var.project_id}/secrets/app-password"
}

import {
  to = google_cloud_run_v2_service.ragbot
  id = "projects/${var.project_id}/locations/${var.region}/services/${var.service_name}"
}

import {
  to = google_cloud_run_v2_service_iam_member.public
  id = "projects/${var.project_id}/locations/${var.region}/services/${var.service_name} roles/run.invoker allUsers"
}

import {
  to = google_project_iam_member.runtime_reads_secrets
  id = "${var.project_id} roles/secretmanager.secretAccessor ${local.runtime_sa}"
}

import {
  to = google_project_iam_member.runtime_builds
  id = "${var.project_id} roles/run.builder ${local.runtime_sa}"
}
