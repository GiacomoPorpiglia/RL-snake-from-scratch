DLL_DIR = dll-files
EXECUTABLE = RLsnake
NATIVE = -march=native
# Default target to build everything
all: $(EXECUTABLE)

# Compile and link the executable
$(EXECUTABLE): compile link copy-dlls

# Compile the C++ files into object files
compile:
	g++ -Icoding/include $(NATIVE) -O3 -c src/*.cpp

# Link the object files to create the final executable
link:
	g++ -O3 *.o -o $(EXECUTABLE) $(NATIVE) -Lcoding/lib -L$(DLL_DIR) -lsfml-graphics -lsfml-window -lsfml-system -lopengl32 -lsfml-audio

# Copy the DLL files from dll-files directory to the current directory
copy-dlls:
	@echo "Copying DLLs from $(DLL_DIR) to the current directory..."
	copy $(DLL_DIR)\*.dll .

# Clean up the project by removing object files and the executable
clean:
	del *.o $(EXECUTABLE)

.PHONY: all compile link copy-dlls clean
