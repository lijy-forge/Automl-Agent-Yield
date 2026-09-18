# Execution Isolation And Safety

Scope: execution guarantees implemented by the current sandbox, process
control, redaction, and task-state code.

## Command construction

Execution tools accept an argument vector instead of a shell command string.
Avoiding `shell=True` prevents shell metacharacters from changing command
meaning. Working directories and environment variables are passed explicitly.

## Container isolation

The Docker sandbox runs with networking disabled and applies CPU, memory, and
process-count limits. The image contains fixed numerical dependencies and does
not bake host API keys into an image layer. Docker execution still requires an
explicit authorization flag.

## Cancellation

Long-running OperationAgent subprocesses start in a separate process group.
Cancellation or timeout sends termination to the group and escalates to a kill
signal after a grace period. This covers descendants created by that operation;
ordinary synchronous Python tools still stop only at their supported boundary.

## Secret handling

Tool arguments, results, and event payloads pass through redaction before
sensitive values are persisted. Authorization headers, API keys, passwords,
database URLs, email addresses, and phone-like values are covered by the
current redaction checks. Redaction reduces exposure but is not encryption.

## Task state

Queued and running tasks are persisted in PostgreSQL in the production path,
while Redis/Celery provides dispatch. Lease and heartbeat fields support
interruption detection and manual recovery. This is not a claim of PostgreSQL
or Redis high availability.
