<?php
// Records one page view per POST from src/visit.js: a JSON line in
// logs/visits-<UTC year-month>.jsonl beside this file. The deploy creates
// logs/ writable by the web server and denies it to HTTP (scripts/deploy-cse.sh),
// and leaves it out of the sync so a deploy keeps it. A view that cannot be
// written is dropped; the page never waits on this.

const MAX_BODY = 2048;
const MAX_FIELD = 300;
const MAX_LOG_BYTES = 200 * 1024 * 1024;

http_response_code(204);
if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    http_response_code(405);
    exit;
}

$body = json_decode((string) file_get_contents('php://input', false, null, 0, MAX_BODY), true);
if (!is_array($body)) {
    exit;
}
$field = static fn ($value) => is_string($value) && $value !== '' ? substr($value, 0, MAX_FIELD) : null;

$path = __DIR__ . '/logs/visits-' . gmdate('Y-m') . '.jsonl';
// A runaway client must not fill the shared web volume.
if (!is_dir(__DIR__ . '/logs') || (is_file($path) && filesize($path) > MAX_LOG_BYTES)) {
    exit;
}
$record = [
    'ts' => gmdate('Y-m-d\TH:i:s\Z'),
    'client' => $_SERVER['REMOTE_ADDR'] ?? null,
    'page' => $field($body['page'] ?? null),
    'referrer' => $field($body['referrer'] ?? null),
    'ua' => $field($_SERVER['HTTP_USER_AGENT'] ?? null),
];
@file_put_contents($path, json_encode($record, JSON_UNESCAPED_SLASHES | JSON_INVALID_UTF8_SUBSTITUTE) . "\n", FILE_APPEND | LOCK_EX);
