SSH_KEY_L = 
SSH_KEY_BG = $(SSH_KEY_L)
USER_L = 
USER_BG = 
VARIANT = 14

PYTHON = python
MPICC = mpiicc
MPICC_FLAGS = -O3
NVCC = nvcc
NVCC_FLAGS = -ccbin $(MPICC) -Xcompiler "$(MPICC_FLAGS) -DCUDA"

TARGET = main


all: clean $(TARGET) $(TARGET)_c $(TARGET)_o

cuda: $(TARGET)_c

omp: $(TARGET)_o

$(TARGET)_c:
	$(NVCC) $(NVCC_FLAGS) main.c main.cu -o $(TARGET)_c -Xcompiler -fopenmp

$(TARGET)_o:
	$(MPICC) $(MPICC_FLAGS) main.c -o $(TARGET)_o -fopenmp

$(TARGET):
	$(MPICC) $(MPICC_FLAGS) main.c -o $(TARGET)

lmount:
	mkdir -p ./lomonosov/
	sudo sshfs -o nonempty -o auto_unmount -oIdentityFile=$(SSH_KEY_L) $(USER_L)@lomonosov.parallel.ru:/mnt/data/users/dm4/vol12/$(USER_L) lomonosov

bgmount:
	mkdir -p ./bluegene/
	sudo sshfs -o nonempty -o auto_unmount -oIdentityFile=$(SSH_KEY_BG) $(USER_BG)@bluegene.hpc.cs.msu.ru:/home/$(USER_BG) bluegene

lupload:
	sudo cp Makefile lomonosov/Makefile
	sudo cp main.c lomonosov/main.c
	sudo cp main.cu lomonosov/main.cu
	sudo cp main.h lomonosov/main.h
	sudo cp $(VARIANT).h lomonosov/variants/$(VARIANT).h
	sudo cp visualize.py lomonosov/visualize.py

bgupload:
	sudo cp Makefile bluegene/Makefile
	sudo cp main.c bluegene/main.c
	sudo cp main.h bluegene/main.h
	sudo cp $(VARIANT).h bluegene/variants/$(VARIANT).h
	sudo cp visualize.py bluegene/visualize.py

umount:
	sudo umount -l bluegene
	sudo umount -l lomonosov

lcopy:
	cp $(TARGET) _scratch/$(TARGET)
	cp $(TARGET)_c _scratch/$(TARGET)_c
	cp $(TARGET)_o _scratch/$(TARGET)_o

clean:
	rm $(TARGET) $(TARGET)_c $(TARGET)_o

bgcustom:
	mpixlc -O3 main.c -o $(TARGET)
	mpixlc_r -qsmp=omp -O3 main.c -o $(TARGET)_o

vis:
	$(PYTHON) visualize.py $(filter-out $@,$(MAKECMDGOALS))

gnuplot:
	gnuplot $(filter-out $@,$(MAKECMDGOALS))

%:
	@:
