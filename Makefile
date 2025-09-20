# Generate a timestamp to use as cache-busting value
CACHE_BUST := $(shell date +%s)

.PHONY: clean scache

clean:
	docker build -t manim .

scache:
	docker build -t manim --build-arg CACHE_BUST=$(CACHE_BUST) .

