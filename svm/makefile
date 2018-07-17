%.o: %.cpp
	g++  $^ -o$@ `pkg-config --cflags --libs opencv`

svm: svm_example.o
	rm *.o && ./svm_example.o

clean:
	rm -rf *.o
