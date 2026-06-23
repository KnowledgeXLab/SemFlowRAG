#!/usr/bin/env bash
set -euo pipefail

DEFAULT_USERNAME="houyuanQ"
DEFAULT_REMOTE="origin"
DEFAULT_REPO_URL="https://github.com/KnowledgeXLab/SemFlowRAG.git"
DEFAULT_CACHE_SECONDS="28800"

usage() {
  cat <<'USAGE'
Usage: scripts/github_token_login.sh [options]

Prompts for a GitHub Personal Access Token (PAT), verifies it, caches it
temporarily for Git, and configures the repository push URL to HTTPS.

Options:
  -u, --username USER      GitHub username. Default: houyuanQ
  -r, --remote NAME        Git remote name. Default: origin
  --repo-url URL           HTTPS repository URL.
                           Default: https://github.com/KnowledgeXLab/SemFlowRAG.git
  --cache-seconds SECONDS  Credential cache duration. Default: 28800
  -h, --help               Show this help.

Security notes:
  - Do not paste your token into chat or shell history.
  - This script reads the token with hidden input.
  - The token is cached in memory by git-credential-cache, not written to disk.
USAGE
}

username="$DEFAULT_USERNAME"
remote="$DEFAULT_REMOTE"
repo_url="$DEFAULT_REPO_URL"
cache_seconds="$DEFAULT_CACHE_SECONDS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--username)
      username="${2:?missing username}"
      shift 2
      ;;
    -r|--remote)
      remote="${2:?missing remote name}"
      shift 2
      ;;
    --repo-url)
      repo_url="${2:?missing repo url}"
      shift 2
      ;;
    --cache-seconds)
      cache_seconds="${2:?missing cache seconds}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run this script inside a Git repository." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required to verify the token." >&2
  exit 1
fi

printf "GitHub username [%s]: " "$username"
read -r input_username
if [[ -n "$input_username" ]]; then
  username="$input_username"
fi

printf "GitHub token: "
stty -echo
read -r token
stty echo
printf "\n"

if [[ -z "$token" ]]; then
  echo "Error: token cannot be empty." >&2
  exit 1
fi

echo "Verifying token with GitHub..."
api_user="$(
  curl -fsS \
    -H "Authorization: Bearer $token" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user |
  sed -n 's/.*"login":[[:space:]]*"\([^"]*\)".*/\1/p' |
  head -n 1
)"

if [[ -z "$api_user" ]]; then
  echo "Error: token verification failed or GitHub user could not be read." >&2
  exit 1
fi

if [[ "$api_user" != "$username" ]]; then
  echo "Warning: token belongs to '$api_user', but username is '$username'." >&2
  username="$api_user"
fi

echo "Configuring Git credential cache for ${cache_seconds}s..."
git config --local credential.helper "cache --timeout=${cache_seconds}"

echo "Saving token to Git credential cache..."
printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' "$username" "$token" |
  git credential approve

echo "Configuring push URL for remote '$remote'..."
git remote set-url --push "$remote" "$repo_url"

echo "Done. You can now try:"
echo "  git push $remote HEAD"
