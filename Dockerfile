FROM ubuntu:latest
MAINTAINER Tang Li Qun "tliqun@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
EXPOSE 5000
WORKDIR /app