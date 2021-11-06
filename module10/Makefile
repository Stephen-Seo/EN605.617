SOURCES = \
	assignment_src/HelloWorld.cpp \
	assignment_src/arg_parse.cc

all: assignment10

assignment10: $(SOURCES)
	$(CXX) $(CXXFLAGS) -g -std=c++11 -o assignment10 -lOpenCL $^

.PHONY: clean

clean:
	rm -f assignment10

# vim: noet
