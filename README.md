# Agentic-Chatbot-using-LangGraph

Agentic Chatbot app cicd deployment with github actions on AWS

Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to docker hub

3. Launch Your EC2

4. Pull Your image from docker hub in EC2

5. Launch your docker image in EC2



3. Create EC2 machine (Ubuntu)

4. Open EC2 and Install docker in EC2 Machine:

    # Optional

    sudo apt-get update -y

    sudo apt-get upgrade

    # Install Docker

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker


    Note: Do the port mapping to this port: - 8501

5. Configure EC2 as self-hosted runner:

    setting>actions>runner>new self hosted runner> choose os> then run command one by one

6. How to add secret keys to GitHub Actions:

REGISTRY=docker.io

DOCKER_USERNAME=

DOCKER_PASSWORD=

IMAGE_NAME=

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION=

OPENAI_API_KEY=

TAVILY_API_KEY=

OPENWEATHER_API_KEY=

LANGSMITH_TRACING=true

LANGSMITH_ENDPOINT=

LANGSMITH_API_KEY=

LANGSMITH_PROJECT=

ALPHA_VANTAGE_API_KEY=

