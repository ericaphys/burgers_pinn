FROM ubuntu:22.04 AS builder-image

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y python3 python3-dev python3-venv python3-pip python3-wheel build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/xpu

FROM ubuntu:22.04 AS runner-image
RUN apt-get update && apt-get install --no-install-recommends -y python3 python3-venv && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home myuser
COPY --from=builder-image /home/myuser/venv /home/myuser/venv

USER myuser
RUN mkdir /home/myuser/code
WORKDIR /home/myuser/code

COPY . .

ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH" 

CMD ["python", "./burgers.py"]

