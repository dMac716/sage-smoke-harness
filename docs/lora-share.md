# Sharing captured LoRa/RF messages with other participants

Our in-dorm Sage node captures over-the-air messages (a participant's
LoRa/433-ISM device) and gives that data back to them — self-serve.

## Path (reuses everything already built)
1. Capture plugin on our node (NEXT BUILD): RTL-SDR -> `rtl_433 -F json`
   (433/868/915 ISM: AcuRite, MeshCore/LoRa-OOK, sensor remotes) -> pywaggle
   `publish("lora.msg", meta={device,model,id,rssi})`. One dongle, no custom RF
   code. (Encrypted LoRaWAN is a separate ChirpStack path — not this.)
2. Beehive is the hand-off: every message is public + queryable, tagged by
   device id. "Providing it to them" = it's already on Beehive.
3. They self-serve with the tools we shipped: fork sage-smoke-harness, set
   HARNESS_PLUGIN to our capture plugin + their device id, and the zero-infra
   Action archives + reports THEIR messages to their own Pages; or
   receiver/serve.py for real-time on a tailnet host.

## Why this shape
- No per-person work at camp: point them at README tiers + their device id.
- Good-citizen: RECEIVE/decode only; never transmit against others' gear.
- Same crash-safe + report + redundancy stack; capture plugin just adds a
  measurement source, consumers unchanged.

## Only new piece
`plugin/lora_capture.py` (rtl_433 JSON -> pywaggle) + a `lora-capture` mode in
launch.sh. Everything downstream already exists and is proven.
