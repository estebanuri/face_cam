# Fixing the v4l2loopback module

This is required to add a virtual video device that can be used by applications like the Chrome, Zoom, Meet...
This script, removes the current, and downloads, builds and installs the module with 
the fix from the official github. 

Follow steps A and B. Step A is required to do once, step B every reboot.

Tested on 
- Ubuntu 18.04
- Linux 5.3.0-46-generic


- step B every time the computer is restarted.

## A) Download the last release from github, build and install


```
# download from git:
cd 
git clone https://github.com/umlaeute/v4l2loopback
cd v4l2loopback

#remove apt package
sudo modprobe -r v4l2loopback
sudo apt remove v4l2loopback-dkms

#install aux
sudo apt-get install linux-generic
sudo apt install dkms

#install v4l2loopback from the repository
https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make

#instal mod
sudo cp -R . /usr/src/v4l2loopback-1.1
sudo dkms add -m v4l2loopback -v 1.1
sudo dkms build -m v4l2loopback -v 1.1
sudo dkms install -m v4l2loopback -v 1.1
sudo reboot
```
Once the computer has rebooted, you'll need to enable the module to be able to use the virtual 
device on the browsers (e. g. Chrome), to do that follow step B.

## B) enable virtual video device 

### B1) enable the module

```
# uninstall the module
modprobe -r v4l2loopback

# list the video devices:
v4l2-ctl --list-devices


	Integrated_Webcam_HD: Integrate (usb-0000:00:14.0-5):
		/dev/video0
		/dev/video1

# the Dummy device is not there

# go to the release directory
cd ~/app/v4l2loopback 

# install the module
sudo modprobe videodev
sudo insmod ./v4l2loopback.ko devices=1 video_nr=2 exclusive_caps=1
sudo depmod -a
sudo modprobe v4l2loopback devices=2

# now list again:
v4l2-ctl --list-devices
Dummy video device (0x0000) (platform:v4l2loopback-000):
	/dev/video2

Integrated_Webcam_HD: Integrate (usb-0000:00:14.0-5):
	/dev/video0
	/dev/video1

# now list again:
v4l2-ctl --list-devices

	Dummy video device (0x0000) (platform:v4l2loopback-000):
		/dev/video2

	Integrated_Webcam_HD: Integrate (usb-0000:00:14.0-5):
		/dev/video0
		/dev/video1


# now the Dummy device IS there :)

```

## run the app
```
python main.py -i 0 -f /dev/video2
```

# now open the site to test if the virtual cam works 
https://webcamtests.com/


If you should be able to see the Dummy video device in the list.



