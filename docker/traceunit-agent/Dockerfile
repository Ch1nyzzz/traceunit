# Role-agent image: codex needs node, and the Test Author / Editor need a
# python toolchain to actually run the tests they are asked to satisfy, plus
# the openai client for live behavioral experiments against the target model.
FROM node:20-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages pytest openai \
    && ln -sf /usr/bin/python3 /usr/local/bin/python
