mkdir linux
FILES=$(cat files.txt)
g++ -std=c++11 $FILES -o linux/regiongrow `pkg-config --cflags --libs opencv4`
#$SHELL