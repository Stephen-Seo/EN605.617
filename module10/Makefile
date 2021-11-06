SOURCES = \
	assignment_src/main.cc \
	assignment_src/arg_parse.cc \
	assignment_src/get_exe_dirname.cc

all: assignment10

assignment10: $(SOURCES)
	$(CXX) $(CXXFLAGS) -g -O0 -std=c++11 -o assignment10 -lOpenCL $^

.PHONY: clean

clean:
	rm -f assignment10

# vim: noet
