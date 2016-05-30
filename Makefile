BUILD=go build
CLEAN=go clean
INSTALL=go install
SRCPATH=./cmd
BUILDPATH=./build

bprop: build
	$(BUILD) -v -o $(BUILDPATH)/bprop $(SRCPATH)/bprop

all: build bprop

install:
	$(INSTALL) $(SRCPATH)/...
clean:
	rm -rf $(BUILDPATH)
build:
	mkdir -p $(BUILDPATH)

.PHONY: clean build
