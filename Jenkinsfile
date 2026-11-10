// Jenkins pipeline: test, build the image, deploy to Kubernetes, smoke-test, roll back on failure.
// Written for a Windows Jenkins agent with Docker Desktop (Kubernetes enabled) and kubectl on the
// PATH. See k8s/README.md for the one-time setup.

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()            // two deploys to one cluster at once would race
    }

    environment {
        IMAGE      = 'rag-pdf-chatbot'
        NAMESPACE  = 'ragbot'
        PROMETHEUS = 'prom/prometheus:v2.54.1'
    }

    stages {
        stage('Unit tests') {
            steps {
                bat '''
                    python -m venv .ci-venv || exit /b 1
                    .ci-venv\\Scripts\\python -m pip install -q -r requirements-dev.txt || exit /b 1
                    .ci-venv\\Scripts\\python -m pytest -q --junitxml=test-results.xml
                '''
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'test-results.xml' }
            }
        }

        stage('Alert rule tests') {
            steps {
                // promtool runs in the official Prometheus image, so nothing extra is installed.
                bat 'docker run --rm -v "%WORKSPACE%\\k8s\\monitoring:/m" -w /m --entrypoint promtool %PROMETHEUS% test rules alerts_test.yml'
                bat 'docker run --rm -v "%WORKSPACE%\\k8s\\monitoring:/m" -w /m --entrypoint promtool %PROMETHEUS% check rules alerts.yml'
            }
        }

        stage('Build image') {
            steps {
                bat 'docker build -t %IMAGE%:%BUILD_NUMBER% -t %IMAGE%:local .'
            }
        }

        stage('Deploy') {
            when { expression { isMainBranch() } }
            steps {
                bat 'kubectl apply -k k8s'
                bat 'kubectl -n %NAMESPACE% set image deployment/ragbot ragbot=%IMAGE%:%BUILD_NUMBER%'
                bat 'kubectl -n %NAMESPACE% rollout status deployment/ragbot --timeout=300s'
            }
            post {
                failure {
                    echo 'Rollout did not become healthy: rolling back to the previous version.'
                    bat 'kubectl -n %NAMESPACE% rollout undo deployment/ragbot'
                }
            }
        }

        stage('Smoke test') {
            when { expression { isMainBranch() } }
            steps {
                // From inside the pod: the app answers its health check and /metrics is served.
                bat 'kubectl -n %NAMESPACE% exec deploy/ragbot -- python -c "import urllib.request as u; assert u.urlopen(\'http://localhost:8080/_stcore/health\', timeout=10).read() == b\'ok\'; assert b\'ragbot_questions_total\' in u.urlopen(\'http://localhost:9100/metrics\', timeout=10).read(); print(\'smoke test passed\')"'
            }
            post {
                failure {
                    echo 'Smoke test failed: rolling back.'
                    bat 'kubectl -n %NAMESPACE% rollout undo deployment/ragbot'
                }
            }
        }
    }

    post {
        success { echo "Deployed ${env.IMAGE}:${env.BUILD_NUMBER}" }
        failure { echo 'Pipeline failed: see the stage logs above.' }
    }
}

// `when { branch 'main' }` only works in multibranch jobs; in a plain pipeline job BRANCH_NAME is
// not set and the deploy stages would be skipped silently. Deploy from main, or from a plain job.
def isMainBranch() {
    return env.BRANCH_NAME == null || env.BRANCH_NAME == 'main'
}
