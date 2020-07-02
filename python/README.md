# Face Swap

*Example*
This example swaps the faces from Alberto Fernandez and Mauricio Macri
```
python face_swap.py -f1 images/alberto_fernandez.webp -f2 images/mauricio_macri.jpg
```


# Video Application

You can use the application and apply effects on screen.

*Example*
This example opens device 0 camera, detects your face and swaps it with Alberto's face

```
python main.py -i 0 -e swap:images/alberto_fernandez.webp
```

If you want to send the output through video conference (e. g. Zoom), 
it is necessary to install a virtual video device. 

## Virtual Video Device

### Linux

For virtual video device I used *v4l2loopback* module. **Important!!! There is a bug on the v4l2loopback and by the time I'm writing this the 
module does not work if you install it directly with apt.** 
Instead you'll need to build and install the last release from the 
official module github [following this instructions](FIX_v4l2loopback.md).

 
*Example*
This example opens device 0 camera, detects your face and swaps it with Alberto's face.
And sends the output to the fake video device /dev/video2. 
*You'll need to have installed a virtual video device.*
Just add the fake argument pointing to the sink virtual video device.

```
python main.py -i 0 -e swap:images/alberto_fernandez.webp -f /dev/video2
```

