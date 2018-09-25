FROM tensorflow/tensorflow:1.10.1-gpu-py3

# Git is required for this benchmark
RUN apt-get update && apt-get install -y --no-install-recommends \
	git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/tensorflow/benchmarks /benchmarks && \
    cd /benchmarks && \
    git checkout 1ef603fd7e568ff75127ec07f160808fcc59911c && \
    rm -rf .git

COPY benchmark.sh /

CMD ["bash", "/benchmark.sh"]
