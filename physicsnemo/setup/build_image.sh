tag=$1
echo $tag
podman-hpc build -t joshuaho/nemo:$tag --platform linux/amd64 .
podman-hpc migrate joshuaho/nemo:$tag
