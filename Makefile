BUILD=go build
CLEAN=go clean
INSTALL=go install
SRCPATH=./cmd
BUILDPATH=./_build
PACKAGES=$(shell go list ./... | grep -v /vendor/)

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
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean build
