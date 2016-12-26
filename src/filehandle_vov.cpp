#include "filehandle_vov.h"

using std::vector;
using std::string;
using std::ostringstream;
using namespace cv;

string NumberToString(int i) {
	ostringstream ss;
	ss << i;
	return ss.str();
}

void writeVoV(FileStorage &f, string name, vvP &contours) {
	f << name;
	f << "{";
	f << "size" << (int)contours.size();
	for (int i = 0, n = contours.size(); i < n; i++) {
		f << name + "_" + NumberToString(i);
		f << contours[i];
	}
	f << "}";
}

void readVoV(FileStorage &f, string name, vvP &contours) {
	FileNode fn = f[name];
	int size = fn["size"];
	contours = vvP(size);
	for (int i = 0; i < size; i++) {
		fn[name + "_" + NumberToString(i)] >> contours[i];
	}
}