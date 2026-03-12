IMAGE_NAME = circuit-dev

build:
	docker build -t $(IMAGE_NAME) .


run:
	@docker run -it --rm \
	 	--gpus all \
		--network host \
		-e DISPLAY=localhost:10.0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $$HOME/.Xauthority:/root/.Xauthority:ro \
		$(IMAGE_NAME)