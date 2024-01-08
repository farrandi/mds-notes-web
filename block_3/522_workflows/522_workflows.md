# Workflows

## Project

Link: [https://github.com/UBC-MDS/english-score-predictor](https://github.com/UBC-MDS/english-score-predictor)

## Docker

### Ten Simple Rules for Writing Dockerfiles

[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008316#sec021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008316#sec021)

1. **Base Your Image on a Minimal Official Docker Image**

   - Use minimal and official base images for simplicity and security.

2. **Tag Images with Explicit Versions**

   - Explicitly tag image versions in `FROM` instructions for reproducibility.

3. **Write Instructions in the Right Order**

   - Optimize Dockerfile instructions order for Docker's layer caching.

4. **Document the Build Context**

   - Clearly document the process to rebuild the image.

5. **Specify Software Versions**

   - Use version pinning for software installations.

6. **Use Version Control**

   - Maintain Dockerfiles in version control systems.

7. **Mount Datasets at Run Time**

   - Store large datasets outside the container and mount them at runtime.

8. **Make the Image One-Click Runnable**

   - Ensure the container is easily runnable with sensible defaults.

9. **Order the Instructions**

   - Order Dockerfile instructions from least to most likely to change.

10. **Regularly Use and Rebuild Containers**

    - Regularly use and update containers to identify and fix issues early.

### Functions

- `docker build -t <image_name> .`: build an image from a Dockerfile
- `docker run --rm -it <image_name> bash`: run a container from an image
  - `-it`: interactive mode
  - `--rm`: remove the container after exiting
- `docker run --rm -it -v <host_dir>:<container_dir> <image_name> bash`: run a container from an image and mount a host directory to a container directory
- `exit`: exit a container

---

- `docker images`: list all images
- `docker rmi <image_name>`: remove an image
- `docker ps -a`: list all containers
- `docker rm <container_name>`: remove a container

- `docker compose up`: run a container from a docker-compose.yml file
  - similar to `conda env create -f environment.yml` and `conda activate <env_name>`

### Dockerfile

- Common structure:
  - `FROM <base_image>`
  - `RUN <command>`
  - `COPY <host_dir> <container_dir>`
  - `WORKDIR <container_dir>`
  - `CMD <command>`

````Dockerfile
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev```
````

`docker compose run --rm <service_name> bash`: run a container from a docker-compose.yml file and mount a host directory to a container directory
