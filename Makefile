build:
	xhost +si:localuser:root
	docker build -t prl_rostorch:noetic-cuda111-torch190 . 

run:
	docker container rm tas || true
	docker run \
		-it \
		-e "DISPLAY" \
		-e "QT_X11_NO_MITSHM=1" \
		-e "XAUTHORITY=${XAUTH}" \
		-v ~/.Xauthority:/root/.Xauthority:rw \
		-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		-v /dev:/dev \
		-v /etc/hosts:/etc/hosts \
		-v /home/youngkyoon/Desktop/DataDisk/dev/MessagePassingFramework/tas_perception/:/ros-ws/src/tas_perception/ \
		--network host \
		--privileged \
		--runtime=nvidia \
		--name tas \
		prl_rostorch:noetic-cuda111-torch190
