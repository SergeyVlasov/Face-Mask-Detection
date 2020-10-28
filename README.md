# Face-Mask-Detection
Detecting face masks using Python, Keras, OpenCV on real video streams

RU version

1) создаем виртуальное кружение

python3 -m virtualenv /path/MyEnv

активируем виртуальное окружение

source /path/MyEnv/bin/activate


устанавливаем все необходимые библиотеки

pip3 install tensorflow
pip3 install keras
pip3 install imutils
pip3 install numpy
pip3 install opencv-python
pip3 install matplotlib
pip3 install scipy


создаем папку для нашего проекта и копируем github 

cd /path/to/project

git clone https://github.com/SergeyVlasov/Face-Mask-Detection


запускаем фаил

python3 detect_mask_video.py



