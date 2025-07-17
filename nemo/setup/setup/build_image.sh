tag=$1
echo $tag
podman-hpc build -t joshuaho/pytorch:$tag --platform linux/amd64 .
podman-hpc migrate joshuaho/pytorch:$tag
