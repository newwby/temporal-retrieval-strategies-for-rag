# docker_bind.ps1 - runs the container from the built image
# run docker_build.ps1 first

Set-Location $PSScriptRoot
docker run --rm -v "${PSScriptRoot}:/app" temporal-retrieval-rag