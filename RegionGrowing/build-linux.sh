mkdir linux
FILES=$(cat files.txt)
g++ -D NDEBUG -std=c++11 $FILES -o linux/RegionGrowing `pkg-config --cflags --libs opencv4`
#$SHELL