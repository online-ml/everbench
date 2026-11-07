# Deploy Everbench on a Beelink server

This deployment runs the Everbench API and worker together in one container. A named Docker volume keeps the SQLite database and weekly Parquet archives across image rebuilds and container replacement.

## Prerequisites

Install [Docker Engine](https://docs.docker.com/engine/install/) and the [Docker Compose plugin](https://docs.docker.com/compose/install/linux/) on the server. Clone this repository on the Beelink, then run the remaining commands from the repository root.

## First deployment

Create the production environment file:

```bash
cp .env.production.example .env.production
MODEL_SIGNING_KEY="$(openssl rand -hex 32)"
API_KEY="$(openssl rand -hex 32)"
sed -i "s/replace-with-a-long-random-secret/$MODEL_SIGNING_KEY/" .env.production
sed -i "s/replace-with-a-separate-long-random-secret/$API_KEY/" .env.production
unset MODEL_SIGNING_KEY API_KEY
chmod 600 .env.production
```

Build and start Everbench:

```bash
docker compose --env-file .env.production up --build --detach
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs --follow --tail 100
```

The service runs migrations and registers configured tasks before it starts the worker and web server. Compose waits up to 30 seconds for in-memory work to drain during a normal stop.

## Access

The default binding is `127.0.0.1:8000` on the Beelink. From another machine, create an SSH tunnel and open <http://localhost:8000>:

```bash
ssh -N -L 8000:127.0.0.1:8000 beelink
```

For direct access on a trusted LAN, set `EVERBENCH_BIND_ADDRESS=0.0.0.0` in `.env.production`, allow TCP port 8000 in the server firewall, and recreate the container:

```bash
docker compose --env-file .env.production up --detach --force-recreate
```

Keep the loopback binding when a reverse proxy or private overlay network provides access. Terminate HTTPS and authentication there if the dashboard will be reachable outside the trusted LAN.

## Update

Back up the data, pull the desired revision, and rebuild:

```bash
git pull --ff-only
docker compose --env-file .env.production up --build --detach
docker compose --env-file .env.production ps
```

Startup applies database migrations before either long-running process begins. Do not run two Everbench containers against the same SQLite volume.

## Backup and restore

Stop Everbench so SQLite checkpoints its write-ahead log, then copy the complete durable volume into a timestamped directory:

```bash
mkdir -p backups
docker compose --env-file .env.production stop everbench
docker compose --env-file .env.production run --rm --no-deps --entrypoint sh \
  -v "$(pwd)/backups:/backup" everbench \
  -c 'destination="/backup/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir "$destination"; cp -a /data/. "$destination/"'
docker compose --env-file .env.production start everbench
```

Copy the `backups` directory to another machine or disk. A backup on the same disk does not protect against disk failure.

To restore, stop Everbench, make a safety copy of the current volume, replace the contents of `/data` from one backup, and start the service. Never combine files from different backup timestamps.

## Operations

Useful checks:

```bash
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs --tail 200 everbench
curl --fail http://127.0.0.1:8000/api/health
docker compose --env-file .env.production exec everbench everbench auto status tasks/wiki_liftwing/task.py
```

The dashboard can run without autonomous research. To enable the weekly research job, add `EVERBENCH_SCHEDULE_RESEARCH=1` and `OPENAI_API_KEY` to `.env.production`, then recreate the container.
