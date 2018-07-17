%.o: %.cpp
	g++  $^ -o $@ `pkg-config --cflags --libs opencv`

single_tracking: single_tracking.o
	./single_tracking.o

svm: svm_example.o
	./svm_example.o

version: version.o
	./version.o

clean:
	rm -rf *.o
