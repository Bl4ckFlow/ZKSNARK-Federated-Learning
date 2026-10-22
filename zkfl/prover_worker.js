/*
 * Persistent Groth16 worker.
 *
 * Spawning `node` once per proof, as the previous implementation did, put
 * roughly a second of interpreter and zkey start-up on the critical path of
 * every round -- which is what the paper's "1-2 s per proof" was mostly
 * measuring.  This process loads the wasm and the proving/verification keys
 * once and then answers newline-delimited JSON commands on stdin:
 *
 *   {"id":1,"op":"prove","input":{...}}      -> {"id":1,"ok":true,"proof":...,"publicSignals":[...]}
 *   {"id":2,"op":"verify","proof":...,"publicSignals":[...]}
 *   {"id":3,"op":"ping"}
 *
 * One line in, one line out, ids echoed, so the Python side can pipeline.
 */
'use strict';

const fs = require('fs');
const readline = require('readline');
const snarkjs = require('snarkjs');

const [wasmPath, zkeyPath, vkeyPath] = process.argv.slice(2);
if (!wasmPath || !zkeyPath || !vkeyPath) {
  process.stderr.write('usage: prover_worker.js <wasm> <zkey> <vkey>\n');
  process.exit(2);
}

let wasm, zkey, vkey;
try {
  wasm = { type: 'mem', data: new Uint8Array(fs.readFileSync(wasmPath)) };
  zkey = { type: 'mem', data: new Uint8Array(fs.readFileSync(zkeyPath)) };
  vkey = JSON.parse(fs.readFileSync(vkeyPath, 'utf8'));
} catch (err) {
  process.stdout.write(JSON.stringify({ id: 0, ok: false, error: `load: ${err.message}` }) + '\n');
  process.exit(3);
}

// snarkjs logs progress to console.log, which would corrupt the protocol.
const logger = { debug() {}, info() {}, warn() {}, error() {} };

const rl = readline.createInterface({ input: process.stdin, terminal: false });
let queue = Promise.resolve();

rl.on('line', (line) => {
  if (!line.trim()) return;
  queue = queue.then(async () => {
    let req;
    try {
      req = JSON.parse(line);
    } catch (err) {
      process.stdout.write(JSON.stringify({ id: null, ok: false, error: 'bad json' }) + '\n');
      return;
    }
    const id = req.id ?? null;
    try {
      if (req.op === 'ping') {
        process.stdout.write(JSON.stringify({ id, ok: true, pong: true }) + '\n');
      } else if (req.op === 'prove') {
        const t0 = process.hrtime.bigint();
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(req.input, wasm, zkey, logger);
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        process.stdout.write(JSON.stringify({ id, ok: true, proof, publicSignals, ms }) + '\n');
      } else if (req.op === 'verify') {
        const t0 = process.hrtime.bigint();
        const valid = await snarkjs.groth16.verify(vkey, req.publicSignals, req.proof, logger);
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        process.stdout.write(JSON.stringify({ id, ok: true, valid, ms }) + '\n');
      } else {
        process.stdout.write(JSON.stringify({ id, ok: false, error: `unknown op ${req.op}` }) + '\n');
      }
    } catch (err) {
      // A witness that violates a constraint lands here; it is a rejected
      // proof, not a crash, so keep serving.
      process.stdout.write(JSON.stringify({ id, ok: false, error: String(err && err.message || err) }) + '\n');
    }
  });
});

rl.on('close', () => { queue.then(() => process.exit(0)); });
