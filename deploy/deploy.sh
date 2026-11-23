#!/usr/bin/env bash
# Deploy Vehicle Tracking container on the cp-bcc VM with candidate verification
# and automatic rollback. Never modifies Nginx or sibling services.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly SCRIPT_DIR DEFAULT_APP_DIR

declare -a DOCKER_RUN_ARGS=()

APP_DIR="${APP_DIR:-${DEFAULT_APP_DIR}}"
BUILD_MODE="${BUILD_MODE:-auto}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
SOURCE_COMMIT="${SOURCE_COMMIT:-}"
IMAGE_REPO="${IMAGE_REPO:-vehicle-tracking}"
CONTAINER_NAME="${CONTAINER_NAME:-vehicle-tracking-demo}"
CANDIDATE_NAME="${CANDIDATE_NAME:-vehicle-tracking-candidate}"
PROD_PORT="${PROD_PORT:-18501}"
CANDIDATE_PORT="${CANDIDATE_PORT:-18502}"
PUBLIC_HOST="${PUBLIC_HOST:-pred-autoscaler.bccdev.id}"
PUBLIC_PORT="${PUBLIC_PORT:-443}"
BASE_URL_PATH="${BASE_URL_PATH:-vehicle-tracking/demo}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-90}"
PUBLIC_DEMO_MODE="${PUBLIC_DEMO_MODE:-1}"
DRY_RUN=0

log_info() {
  printf '[deploy] %s\n' "$*"
}

log_error() {
  printf '[deploy][ERROR] %s\n' "$*" >&2
}

usage() {
  cat <<'EOF'
Usage: deploy/deploy.sh [--dry-run] [--help]

Build and roll out the Vehicle Tracking Streamlit container on the VM:
  1. Build a candidate image (Dockerfile.update when a tested runtime image exists,
     or Dockerfile for a full build).
  2. Start an isolated candidate container on 127.0.0.1:18502 and verify health,
     root page response, and packaged UI editor assets.
  3. Stop and back up the active production container, then start the new
     production container on 127.0.0.1:18501 with the same bind mounts and
     resource limits.
  4. Roll back automatically to the backup container if production health fails.

Environment variables:
  APP_DIR                 Repository checkout path on the VM (default: repo root)
  BUILD_MODE              auto | update | full (default: auto)
  RUNTIME_IMAGE           Base image for Dockerfile.update (auto-detected if empty)
  SOURCE_COMMIT           Git commit SHA label (default: git rev-parse HEAD)
  IMAGE_REPO              Docker image repository name (default: vehicle-tracking)
  CONTAINER_NAME          Production container name (default: vehicle-tracking-demo)
  CANDIDATE_NAME          Temporary candidate container (default: vehicle-tracking-candidate)
  PROD_PORT               Host loopback port for prod (default: 18501)
  CANDIDATE_PORT          Host loopback port for candidate (default: 18502)
  PUBLIC_HOST             Streamlit browser server address (default: pred-autoscaler.bccdev.id)
  PUBLIC_PORT             Streamlit browser server port (default: 443)
  BASE_URL_PATH           URL prefix without leading slash (default: vehicle-tracking/demo)
  HEALTH_TIMEOUT_SECONDS  Seconds to wait for healthcheck (default: 90)
  PUBLIC_DEMO_MODE        VEHICLE_COUNTER_PUBLIC_DEMO value (default: 1)
EOF
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --dry-run)
        DRY_RUN=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log_error "Unknown argument: $1"
        usage >&2
        exit 2
        ;;
    esac
    shift
  done
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_error "Required command not found: ${cmd}"
    exit 1
  fi
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || ((value <= 0)); then
    log_error "${name} must be a positive integer, got: ${value}"
    exit 1
  fi
}

validate_inputs() {
  case "${BUILD_MODE}" in
    auto|update|full) ;;
    *)
      log_error "BUILD_MODE must be one of auto, update, full; got: ${BUILD_MODE}"
      exit 1
      ;;
  esac

  validate_positive_int "PROD_PORT" "${PROD_PORT}"
  validate_positive_int "CANDIDATE_PORT" "${CANDIDATE_PORT}"
  validate_positive_int "PUBLIC_PORT" "${PUBLIC_PORT}"
  validate_positive_int "HEALTH_TIMEOUT_SECONDS" "${HEALTH_TIMEOUT_SECONDS}"

  if [[ "${PROD_PORT}" == "${CANDIDATE_PORT}" ]]; then
    log_error "PROD_PORT and CANDIDATE_PORT must differ (${PROD_PORT})"
    exit 1
  fi

  if [[ ! -d "${APP_DIR}" ]]; then
    log_error "APP_DIR does not exist: ${APP_DIR}"
    exit 1
  fi
}

cleanup_candidate() {
  if ((DRY_RUN == 1)); then
    return 0
  fi
  if command -v docker >/dev/null 2>&1 && docker container inspect "${CANDIDATE_NAME}" >/dev/null 2>&1; then
    docker rm -f "${CANDIDATE_NAME}" >/dev/null 2>&1 || true
  fi
}

resolve_source_commit() {
  if [[ -n "${SOURCE_COMMIT}" ]]; then
    printf '%s\n' "${SOURCE_COMMIT}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 && git -C "${APP_DIR}" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "${APP_DIR}" rev-parse HEAD
    return 0
  fi
  printf 'unknown\n'
}

detect_runtime_image() {
  if [[ -n "${RUNTIME_IMAGE}" ]]; then
    printf '%s\n' "${RUNTIME_IMAGE}"
    return 0
  fi
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker inspect --format '{{.Config.Image}}' "${CONTAINER_NAME}" 2>/dev/null || true
    return 0
  fi
  printf '\n'
}

wait_for_health() {
  local port="$1"
  local container="$2"
  local health_url="http://127.0.0.1:${port}/${BASE_URL_PATH}/_stcore/health"
  local elapsed=0

  while ((elapsed < HEALTH_TIMEOUT_SECONDS)); do
    local state
    state="$(docker inspect --format '{{.State.Status}}' "${container}" 2>/dev/null || printf 'missing')"
    if [[ "${state}" != "running" && "${state}" != "created" ]]; then
      log_error "Container ${container} exited early (state=${state})"
      return 1
    fi

    if curl --fail --silent --show-error --max-time 4 "${health_url}" >/dev/null 2>&1; then
      log_info "Healthcheck passed on ${health_url}"
      return 0
    fi

    sleep 2
    elapsed=$((elapsed + 2))
  done

  log_error "Timed out waiting for ${health_url} after ${HEALTH_TIMEOUT_SECONDS}s"
  return 1
}

build_docker_run_args() {
  local name="$1"
  local port="$2"
  local restart_policy="$3"
  local image="$4"

  DOCKER_RUN_ARGS=(
    run -d
    --name "${name}"
    --restart "${restart_policy}"
    --cpus=1.25
    --memory=1400m
    --memory-swap=1400m
    --pids-limit=256
    --read-only
    --tmpfs "/tmp:size=160m,mode=1777"
    --cap-drop=ALL
    --security-opt=no-new-privileges
    -e "STREAMLIT_BROWSER_SERVER_ADDRESS=${PUBLIC_HOST}"
    -e "STREAMLIT_BROWSER_SERVER_PORT=${PUBLIC_PORT}"
    -e "VEHICLE_COUNTER_PUBLIC_DEMO=${PUBLIC_DEMO_MODE}"
    -p "127.0.0.1:${port}:8501"
    -v "${APP_DIR}/models:/app/models:ro"
    -v "${APP_DIR}/data/videos:/app/data/videos:ro"
    -v "${APP_DIR}/outputs:/app/outputs"
    "${image}"
  )
}

verify_candidate() {
  local port="$1"
  local root_url="http://127.0.0.1:${port}/${BASE_URL_PATH}/"

  wait_for_health "${port}" "${CANDIDATE_NAME}"

  if ! curl --fail --silent --show-error --max-time 8 "${root_url}" >/dev/null; then
    log_error "Candidate root page check failed on ${root_url}"
    return 1
  fi

  docker exec "${CANDIDATE_NAME}" python -c '
import pathlib
import vehicle_counter.ui as ui
import vehicle_counter.ui.layout

base = pathlib.Path(ui.__file__).parent
for rel in ("editor/index.html", "editor/shapes/index.html"):
    path = base / rel
    assert path.is_file(), f"missing {path}"
'
  log_info "Candidate UI package and editor assets verified"
}

main() {
  parse_args "$@"
  validate_inputs

  local resolved_commit short_sha target_image effective_mode detected_runtime=""
  resolved_commit="$(resolve_source_commit)"
  short_sha="${resolved_commit:0:12}"
  short_sha="${short_sha//[^a-zA-Z0-9._-]/}"
  if [[ -z "${short_sha}" ]]; then
    short_sha="manual"
  fi
  target_image="${IMAGE_REPO}:${short_sha}"

  if ((DRY_RUN == 1)); then
    log_info "Dry run enabled; no containers or images will be modified"
    log_info "APP_DIR=${APP_DIR}"
    log_info "BUILD_MODE=${BUILD_MODE}"
    log_info "SOURCE_COMMIT=${resolved_commit}"
    log_info "TARGET_IMAGE=${target_image}"
    log_info "CONTAINER_NAME=${CONTAINER_NAME} (127.0.0.1:${PROD_PORT}:8501)"
    log_info "CANDIDATE_NAME=${CANDIDATE_NAME} (127.0.0.1:${CANDIDATE_PORT}:8501)"
    exit 0
  fi

  require_command docker
  require_command curl

  local required_dir
  for required_dir in models data/videos outputs; do
    if [[ ! -d "${APP_DIR}/${required_dir}" ]]; then
      log_error "Required runtime directory is missing: ${APP_DIR}/${required_dir}"
      exit 1
    fi
  done

  trap cleanup_candidate EXIT

  if [[ "${BUILD_MODE}" == "full" ]]; then
    effective_mode="full"
  else
    detected_runtime="$(detect_runtime_image)"
    if [[ -n "${detected_runtime}" ]] && docker image inspect "${detected_runtime}" >/dev/null 2>&1; then
      effective_mode="update"
    elif [[ "${BUILD_MODE}" == "update" ]]; then
      log_error "BUILD_MODE=update requires an existing local RUNTIME_IMAGE (detected: '${detected_runtime}')"
      exit 1
    else
      effective_mode="full"
    fi
  fi

  log_info "Building ${target_image} (mode=${effective_mode}, commit=${resolved_commit})"
  if [[ "${effective_mode}" == "update" ]]; then
    log_info "Reusing runtime base image: ${detected_runtime}"
    docker build \
      -f "${APP_DIR}/deploy/Dockerfile.update" \
      --build-arg "RUNTIME_IMAGE=${detected_runtime}" \
      --build-arg "SOURCE_COMMIT=${resolved_commit}" \
      -t "${target_image}" \
      "${APP_DIR}"
  else
    docker build \
      -f "${APP_DIR}/deploy/Dockerfile" \
      -t "${target_image}" \
      "${APP_DIR}"
  fi

  cleanup_candidate

  log_info "Starting candidate container ${CANDIDATE_NAME} on 127.0.0.1:${CANDIDATE_PORT}"
  build_docker_run_args "${CANDIDATE_NAME}" "${CANDIDATE_PORT}" "no" "${target_image}"
  docker "${DOCKER_RUN_ARGS[@]}"

  if ! verify_candidate "${CANDIDATE_PORT}"; then
    log_error "Candidate verification failed; production container was not modified"
    docker logs --tail 50 "${CANDIDATE_NAME}" >&2 || true
    exit 1
  fi

  log_info "Removing verified candidate container before production cutover"
  cleanup_candidate

  local backup_name=""
  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    backup_name="vehicle-tracking-backup-${timestamp}-${short_sha}"
    log_info "Stopping current ${CONTAINER_NAME} and preserving backup as ${backup_name}"
    docker stop --time 15 "${CONTAINER_NAME}" >/dev/null || true
    docker rename "${CONTAINER_NAME}" "${backup_name}"
  fi

  log_info "Starting production container ${CONTAINER_NAME} on 127.0.0.1:${PROD_PORT}"
  build_docker_run_args "${CONTAINER_NAME}" "${PROD_PORT}" "unless-stopped" "${target_image}"
  if ! docker "${DOCKER_RUN_ARGS[@]}" || ! wait_for_health "${PROD_PORT}" "${CONTAINER_NAME}"; then
    log_error "Production startup failed; rolling back"
    docker logs --tail 50 "${CONTAINER_NAME}" >&2 || true
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    if [[ -n "${backup_name}" ]] && docker container inspect "${backup_name}" >/dev/null 2>&1; then
      docker rename "${backup_name}" "${CONTAINER_NAME}"
      docker start "${CONTAINER_NAME}" >/dev/null
      if wait_for_health "${PROD_PORT}" "${CONTAINER_NAME}"; then
        log_info "Rollback succeeded; restored previous ${CONTAINER_NAME}"
      else
        log_error "Rollback container started but failed healthcheck"
      fi
    fi
    exit 1
  fi

  local container_id
  container_id="$(docker inspect --format '{{.Id}}' "${CONTAINER_NAME}")"
  log_info "Deployment succeeded: image=${target_image} container=${container_id:0:12}"
  if [[ -n "${backup_name}" ]]; then
    log_info "Previous container kept stopped for rollback: ${backup_name}"
  fi
  log_info "Nginx and other VM services were not modified"
}

main "$@"
