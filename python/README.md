# Face Swap

*Example*
python face_swap.py -f1 images/alberto_fernandez.webp -f2 images/mauricio_macri.jpg

# Video Application

You can use the application and apply effects on screen.

*Example*
This example opens device 0 camera, detects your face and swaps it with Alberto's face

```
python main.py -i 0 -e swap:images/alberto_fernandez.webp
```

If you want to send the output through video conference (e. g. Zoom), 
it is necessary to install a virtual video device. 

##Virtual Video Device
###Linux

For virtual video device I used *v4l2loopback* module.
 
Tested on 
- Ubuntu 18.04
- Linux 5.3.0-46-generic

**Important!!!**
There is a bug on the v4l2loopback and by the time I'm writing this the 
module **does not work** on Ubuntu 18.04 if you install it with apt.
Instead follow the instructions explained in *Fixing the v4l2loopback module* 

*Example*
This example opens device 0 camera, detects your face and swaps it with Alberto's face.
And sends the output to the fake video device /dev/video2. 
*You'll need to have installed a virtual video device.*
Just add the fake argument pointing to the sink virtual video device.

```
python main.py -i 0 -e swap:images/alberto_fernandez.webp -f /dev/video2
```


##Fixing the v4l2loopback module
This script, removes the current, and downloads, builds and installs the module with 
the fix from the official github. 

```
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
device on the browsers (e. g. Chrome), to do that do:


```
modinfo v4l2loopback
```
```
filename:       /lib/modules/5.3.0-61-generic/extra/v4l2loopback.ko
...
```


This will list a lot of info. Check where the .ko file is pointing.
Then run the following commands

```
sudo insmod /lib/modules/5.3.0-61-generic/extra/v4l2loopback.ko devices=1 video_nr=2 exclusive_caps=1
sudo depmod -a
sudo modprobe v4l2loopback devices=2
```

If you should be able to list the video devices and see a Dummy video device for video loopback:

```
v4l2-ctl --list-devices
```

```
Dummy video device (0x0000) (platform:v4l2loopback-000):
	/dev/video2

Integrated_Webcam_HD: Integrate (usb-0000:00:14.0-5):
	/dev/video0
	/dev/video1

```

