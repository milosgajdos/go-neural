BUILD=go build
CLEAN=go clean
INSTALL=go install
SRCPATH=./cmd
BUILDPATH=./_build

bprop: test build
	$(BUILD) -v -o $(BUILDPATH)/bprop $(SRCPATH)/bprop

all: build bprop

install:
	$(INSTALL) $(SRCPATH)/...
clean:
	rm -rf $(BUILDPATH)
build:
	mkdir -p $(BUILDPATH)
test:
	go test -cover $$(go list ./... | grep -v /vendor/)

.PHONY: clean build
