
#!/bin/bash

make scache

docker run -it --rm -p 8888:8888 manim \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root