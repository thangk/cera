FROM alpine:latest

ARG PB_VERSION=0.36.2

RUN apk add --no-cache unzip ca-certificates

ADD https://github.com/pocketbase/pocketbase/releases/download/v${PB_VERSION}/pocketbase_${PB_VERSION}_linux_amd64.zip /tmp/pb.zip
RUN unzip /tmp/pb.zip -d /pb/

COPY ./pb_migrations /pb/pb_migrations

EXPOSE 8090

# Create default superuser on first boot, then start the server
CMD /pb/pocketbase superuser upsert admin@cera.local adminadmin 2>/dev/null; \
    /pb/pocketbase serve --http=0.0.0.0:8090
