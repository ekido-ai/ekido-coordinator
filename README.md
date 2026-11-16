# ekido-coordinator

Schedules training jobs across devices: placement, recovery when a device drops, checkpoints, quality checks and settlement. FastAPI on 127.0.0.1:8003; its tables live in the `coordinator` schema.

Part of [Ekido](https://github.com/ekido-ai): your own small AI model, trained on spare devices and paid for only when it works. The shared library, the API contract and the evidence behind every number live in [ekido-core](https://github.com/ekido-ai/ekido-core).

## Rules

- Every change is a pull request with the full report (the PR template); the **Gate** check refuses it otherwise.
- Testnet and test money only. Never commit keys, `.env` contents, wallet keys or customer data.
- Every number we quote is measured and has its raw file.

Licence: all rights reserved (see LICENSE).
