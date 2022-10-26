MOUNT_WORKING_VOLUME = -it -v ${CURDIR}:/w -w /w

all: bash
.PHONY: all builder

# build docker image with necessary installations
docker: build/.builder

# open the bash in a container with the files mounted in the
# working volume
bash: builder FORCE
	docker run $(MOUNT_WORKING_VOLUME) \
			dttt:latest bash

FORCE:

build/.build:
		mkdir -p build
		touch build/.build

build/.builder: build/.build Dockerfile
		docker build \
				-t dttt:latest \
				-f Dockerfile \
				.
		touch build/.builder
