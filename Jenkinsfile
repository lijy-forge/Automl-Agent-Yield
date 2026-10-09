pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }

    parameters {
        string(
            name: 'BRANCH',
            defaultValue: 'jenkins-lab',
            description: '要构建的 Git 分支'
        )
        choice(
            name: 'ENVIRONMENT',
            choices: ['dev', 'staging', 'production'],
            description: '模拟部署环境'
        )
    }

    stages {
        stage('Checkout') {
            steps {
                deleteDir()
                git branch: params.BRANCH,
                    url: 'https://github.com/lijy-forge/Automl-Agent-Yield.git'

                echo "Building branch=${params.BRANCH}, environment=${params.ENVIRONMENT}"

                sh '''
                    set -eux
                    python3 --version
                    python3 -m venv .venv-ci
                    .venv-ci/bin/python -m pip install --upgrade pip
                    .venv-ci/bin/pip install -r requirements-ci.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    set -eux
                    .venv-ci/bin/ruff check \
                        yieldmind/model_usage.py \
                        tests/test_model_usage.py
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    set -eux
                    mkdir -p reports
                    .venv-ci/bin/python -m pytest -q \
                        tests/test_model_usage.py \
                        --junitxml=reports/pytest.xml
                '''
            }
        }

        stage('Build') {
            steps {
                sh '''
                    set -eux
                    mkdir -p dist
                    tar -czf "dist/automl-agent-yield-${BUILD_NUMBER}.tar.gz" \
                        yieldmind/model_usage.py \
                        tests/test_model_usage.py \
                        README.md
                    sha256sum "dist/automl-agent-yield-${BUILD_NUMBER}.tar.gz"
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/pytest.xml'
        }

        success {
            archiveArtifacts artifacts: 'dist/*.tar.gz', fingerprint: true
        }
    }
}
