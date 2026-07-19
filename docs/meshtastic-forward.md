# Forwarding Meshtastic mesh data to other participants

Bring MeshCore devices, flash one (or more) to **Meshtastic**. One attached to
our dorm Sage node = a **gateway**: a mesh propagates every node's packets, so
our gateway receives ALL participants' traffic on the shared channel and
forwards it to Beehive — they self-serve their own node's data.

## Path (downstream is 100% reuse)
1. Gateway plugin on our node (NEW BUILD, small): one Meshtastic device on USB
   -> `meshtastic` Python lib (serial/BLE, protobuf API) -> pywaggle
   `publish("mesh.msg"/"mesh.telemetry"/"mesh.position",
   meta={from_id,to_id,channel,snr,rssi,portnum})`. Clean API — no SDR/RF
   decode. (Contrast: rtl_433 path = for non-Meshtastic 433-ISM devices.)
2. Beehive = hand-off: every packet public + queryable by node id.
3. Others self-serve with shipped tools: fork sage-smoke-harness, filter their
   node id -> Action archives + Pages report; or receiver/serve.py real-time.

## INTEROP requirement (tell participants once)
Their device must share our mesh: same **region (US915)**, same **channel
name + PSK**, compatible Meshtastic firmware. Publish the channel QR/URL once;
anyone who joins is auto-forwarded. No per-person wiring.

## Good-citizen
Shared mesh — RECEIVE + forward only. Don't spoof/DM others' nodes; don't
bridge private channels without consent. Position data is sensitive — forward
only the default/agreed channel.

## Only new piece
`plugin/mesh_gateway.py` (meshtastic lib -> pywaggle) + `mesh-gateway` mode in
launch.sh. Beehive, archiver, report, receiver, 3-tier redundancy all exist.
