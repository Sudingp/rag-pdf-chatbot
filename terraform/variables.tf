variable "project_id" {
  description = "Google Cloud project that runs the chatbot"
  type        = string
}

variable "region" {
  description = "Cloud Run region"
  type        = string
  default     = "asia-south1"
}

variable "service_name" {
  type    = string
  default = "rag-pdf-chatbot"
}

variable "image" {
  description = "Container image. Deploys change it (gcloud or CI), so Terraform does not fight over it."
  type        = string
  default     = "asia-south1-docker.pkg.dev/PROJECT/cloud-run-source-deploy/rag-pdf-chatbot"
}
