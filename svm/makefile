%.o: %.cpp
	g++  $^ -o$@ `pkg-config --cflags --libs opencv`

svm: svm_example.o
	./svm_example.o

train: train.o
	./train.o

trainHog: train_HOG.o
	./train_HOG.o -dw=64 -dh=128 -pd=/home/nam/data/data_khanh_4k5/single_img_resize/motorbike -nd=/home/nam/data/data_khanh_4k5/single_img_resize/bicycle -fn=HOGpedestrian64x128.xml
train_resize: train_image.o
	./train_image.o -dw=64 -dh=128 -pd=/home/nam/data/data_khanh_4k5/single_img_resize/motorbike -nd=/home/nam/data/data_khanh_4k5/single_img_resize/bicycle -fn=HOGpedestrian64x128.xml

train_multiple: train_multiple_class.o
	./train_multiple_class.o


test_multiple: test_multiple_class.o
	./test_multiple_class.o

testHOG: test_HOG_SVM.o
	./test_HOG_SVM.o


clean:
	rm -rf *.o
