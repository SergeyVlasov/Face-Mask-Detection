# Face-Mask-Detection
Определение наличия маски и (в зависимости от маски) открытие двери

![til](https://github.com/SergeyVlasov/Face-Mask-Detection/blob/master/media/mask_detect.gif)





Raspberry Pi 3 B+

![Image alt](https://github.com/SergeyVlasov/Face-Mask-Detection/blob/master/media/raspberry.jpg)

step motor 28BYJ-48

![Image alt](https://github.com/SergeyVlasov/Face-Mask-Detection/blob/master/media/28BYJ-48.jpg)


motor driver ULN2003

![Image alt](https://github.com/SergeyVlasov/Face-Mask-Detection/blob/master/media/ULN2003.jpg)


подключение драйвера


![Image alt](https://github.com/SergeyVlasov/Face-Mask-Detection/blob/master/media/pin.jpg)


Установка:

1) создаем виртуальное кружение

- python3 -m virtualenv /path/MyEnv

2) активируем виртуальное окружение

- source /path/MyEnv/bin/activate


3) устанавливаем все необходимые библиотеки

- pip3 install tensorflow
- pip3 install keras
- pip3 install imutils
- pip3 install numpy
- pip3 install opencv-python
- pip3 install matplotlib
- pip3 install scipy


4) создаем папку для нашего проекта и копируем github 

- cd /path/to/project

- git clone https://github.com/SergeyVlasov/Face-Mask-Detection


5) запускаем фаил

- python3 detect_mask_video.py



