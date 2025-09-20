.PHONY: clean scache

clean:
	docker build -t manim .

scache:
	docker build -t manim --build-arg CACHE_BUST=$(shell date +%s) .

