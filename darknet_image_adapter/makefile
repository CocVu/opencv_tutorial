CPP=g++
OBJDIR=./obj/
	VPATH=./src/

EXEC= crop_image

COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv`

OBJ=crop.o blob.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP)  -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@

$(EXEC): $(OBJS)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

all: obj output $(EXEC)

run: $(EXEC)
	./$(EXEC)

obj:
	mkdir -p obj

output:
	mkdir -p output
