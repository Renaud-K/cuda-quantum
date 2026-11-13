# CUDA-Q Realtime Architecture

CUDA-Q Realtime is an RPC system over RDMA between a quantum processor's
system controller (the `QSC`, typically an FPGA) and a host. The `QSC` sends a
request, a handler on the host's CPU or GPU runs it within microseconds, and
the response goes back the same way.

This page is the short tour, mostly pictures. The pages it links to hold the
details.

## Vocabulary

| Term | Meaning | In the code |
|------|---------|-------------|
| Data path | Where a request lands and which processor dispatches it: the CPU path or the GPU path | `CUDAQ_DISPATCH_PATH_HOST`, `CUDAQ_DISPATCH_PATH_DEVICE` |
| Dispatcher | The loop that reads a request, finds its handler, runs it and writes the response. A CPU dispatcher is a thread, a GPU dispatcher a persistent kernel | `cudaq_dispatcher_t` |
| Handler | The function that serves one RPC, registered in the function table under the hash of its name | `cudaq_function_entry_t` |
| Dispatch mode | How the dispatcher calls a handler | `CUDAQ_DISPATCH_DEVICE_CALL`, `CUDAQ_DISPATCH_HOST_CALL`, `CUDAQ_DISPATCH_GRAPH_LAUNCH` |
| Bridge | A plug-in library that implements the bridge API | `cudaq_realtime_bridge_interface_t` |
| Transport provider | The networking stack under a bridge: DOCA `GPUNetIO`, `libibverbs`, sockets, or a partner's stack | |
| Rings | The RX and TX slot arrays requests and responses live in | `cudaq_ringbuffer_t` |
| Three-worker model | An RX worker, the dispatcher and a TX worker loop independently and share only the rings | `CUDAQ_KERNEL_REGULAR` |
| Unified worker | The dispatcher is the only loop and drives the transport through the bridge's data plane | `CUDAQ_KERNEL_UNIFIED` |
| Data plane | The `rx_poll` and `tx_publish` hooks a bridge hands the unified worker | `cudaq_cpu_dataplane_t`, `cudaq_gpu_dataplane_t` |

The three-worker model is what older pages call the "3-kernel architecture".
It works the same way with CPU threads, hence the name used here.

## RPC over RDMA

The `QSC` and the host each own a set of rings. At connect time the `QSC`
learns where the host's RX ring is (base address, slot stride, remote key), so
it can write a request straight into any slot. No socket and no CPU copy sit
on the data path: the NIC writes into ring memory and the dispatcher reads it
there.

## Two data paths

Requests take one of two paths, chosen by where the rings live:

- **GPU path**: the NIC writes into rings in GPU memory, and a persistent
  dispatcher kernel runs `DEVICE_CALL` handlers on the GPU. The CPU launches
  that kernel once and stays off the data path. Served by the GPU RoCE bridge.
- **CPU path**: the NIC writes into rings in host memory, and a dispatcher
  thread runs `HOST_CALL` handlers on the CPU. Served by the CPU RoCE and `UDP`
  bridges.

```{mermaid}
flowchart LR
    qsc["QSC<br/>QPU system controller, FPGA"]
    subgraph gpuPath [GPU path]
        direction LR
        nicG["NIC"]
        ringsG["RX and TX rings<br/>in GPU memory"]
        dispG["GPU dispatcher<br/>persistent kernel"]
        handlersG["DEVICE_CALL handlers"]
        launchG["CPU<br/>launches the dispatcher once"]
        nicG <--> ringsG
        ringsG <--> dispG
        dispG --> handlersG
        launchG -.-> dispG
    end
    subgraph cpuPath [CPU path]
        direction LR
        nicC["NIC"]
        ringsC["RX and TX rings<br/>in host memory"]
        dispC["CPU dispatcher<br/>thread"]
        handlersC["HOST_CALL handlers"]
        nicC <--> ringsC
        ringsC <--> dispC
        dispC --> handlersC
    end
    qsc <-->|"RDMA"| nicC
    qsc <-->|"RDMA"| nicG
```

The two paths do not talk to each other. A GPU dispatcher and a CPU dispatcher
can share one ring set (`shared_ring_mode`), each serving the function IDs in
its own table, but neither hands a request to the other.

### From the CPU path to the GPU

A request that reaches the CPU can still use the GPU: its `HOST_CALL` handler
copies the data to GPU memory and launches a kernel, or hands the request to a
kernel that is already running, which may do its own dispatch. The CPU
dispatcher publishes the response when the handler returns, so the handler
waits for the GPU work first. The dispatcher thread has no CUDA device
selected; a handler that uses CUDA selects its own.

Example: in review as
[#5878](https://github.com/NVIDIA/cuda-quantum/pull/5878). A `HOST_CALL`
handler over the CPU RoCE bridge hands each request through a mailbox to a
cooperative grid launched once at startup, and waits for the grid's reply. Link
its page here once it is merged.

The CPU dispatcher does not launch CUDA graphs itself: `GRAPH_LAUNCH` entries
belong to the GPU path.

## Two layers, two roles

Realtime has two layers: the **dispatcher**, which runs handlers, and the
**transport**, which moves bytes. The bridge API between them makes realtime
transport-agnostic: every transport is a plug-in behind the same API, and any
application on top works with any of them.

```{mermaid}
flowchart TB
    subgraph appLayer [Applications]
        direction TB
        qec["QEC decoding server"]
        calib["Calibration"]
        custom["Partner handlers"]
        appDev(["Role: application developer<br/>writes handlers, picks a bridge"])
    end
    subgraph realtime [CUDA-Q Realtime]
        direction TB
        subgraph dispLayer [Dispatcher layer]
            direction TB
            gpuDisp["GPU dispatcher<br/>persistent kernel, DEVICE_CALL"]
            cpuDisp["CPU dispatcher<br/>thread, HOST_CALL"]
        end
        subgraph transportLayer [Transport layer: bridge plug-ins]
            direction TB
            gpuRoce["GPU RoCE<br/>DOCA GPUNetIO"]
            cpuRoce["CPU RoCE<br/>libibverbs"]
            udp["UDP<br/>sockets"]
            partner["Partner bridge<br/>vendor stack"]
            provider(["Role: transport provider<br/>implements the bridge API"])
        end
        dispLayer <-->|"bridge API, bridge_interface.h"| transportLayer
    end
    wire["Wire to the QSC"]
    appLayer <-->|"dispatcher API, cudaq_realtime.h"| dispLayer
    transportLayer <--> wire
```

The two APIs give two roles:

- **Transport provider: bring your own transport.** Implement the bridge API in
  [`bridge_interface.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/bridge/bridge_interface.h)
  and ship it as a plug-in library. Existing applications, such as the CUDA-Q
  QEC decoding server, then run over it unchanged. See the
  [network interface page](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/cudaq_realtime_network_interface.md)
  for the walkthrough.
- **Application developer: bring your own handlers.** Program against the
  dispatcher API in
  [`cudaq_realtime.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h):
  register a function table of handlers and run a dispatcher. Choosing the
  bridge library and wiring it to the dispatcher is the application's job.

Wiring a dispatcher to a bridge takes the same few calls for every transport,
data path and worker model:

```c
cudaq_bridge_create_from_library(&bridge, "libcudaq-realtime-bridge-udp.so",
                                 argc, argv);           // load the plug-in
cudaq_dispatch_manager_create(&manager);
cudaq_dispatcher_create(manager, &config, &dispatcher); // path, mode, model
cudaq_dispatcher_set_bridge(dispatcher, bridge);        // rings or data plane
cudaq_dispatcher_set_function_table(dispatcher, &table);
cudaq_dispatcher_set_control(dispatcher, &shutdown_flag, &stats);
cudaq_dispatcher_start(dispatcher);
cudaq_bridge_connect(bridge);
cudaq_bridge_launch(bridge);
```

Examples, in the order to read them:

- [`cudaq_realtime_test_server.cpp`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/unittests/test_server/cudaq_realtime_test_server.cpp):
  one server for any bridge library, either data path and either worker model,
  with each wiring step numbered.
- [`test_dispatcher_set_bridge.cpp`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/unittests/bridge_interface/udp/test_dispatcher_set_bridge.cpp):
  what `cudaq_dispatcher_set_bridge` accepts and refuses, over the `UDP` bridge.
- [`roce_app.cpp`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/validation/roce_app.cpp):
  one receiver for both RoCE bridges, on the CPU or the GPU path, as used for
  the FPGA validation runs.

## The four contracts

The whole architecture comes down to four contracts. Each one is the boundary
between two parties, and each can change on one side without the other as
long as the contract holds.

```{mermaid}
flowchart TB
    app["Application"]
    disp["Dispatcher"]
    handler["Handlers"]
    bridge["Bridge plug-in"]
    qsc["QSC"]
    app -->|"dispatcher API<br/>create, wire, start, stop"| disp
    disp -->|"handler calling convention<br/>rx_slot, tx_slot, slot_size"| handler
    disp <-->|"bridge API<br/>rings, data plane hooks"| bridge
    bridge <-->|"RPC protocol<br/>what every ring slot holds"| qsc
```

| Contract | Between | Defined in | Read more |
|----------|---------|------------|-----------|
| Dispatcher API | application and dispatcher | [`cudaq_realtime.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h) | [Host API](host.md) |
| Bridge API | dispatcher and transport | [`bridge_interface.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/bridge/bridge_interface.h) | [Network interface](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/cudaq_realtime_network_interface.md#extending-cuda-q-realtime-with-a-custom-networking-interface) |
| RPC protocol | `QSC` and handler, through the ring slots | [`dispatch_kernel_launch.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h) | [Message protocol](protocol.md) |
| Handler calling convention | dispatcher and handler | `DeviceRPCFunction` in [`dispatch_kernel_launch.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h), `cudaq_host_rpc_fn_t` in [`cudaq_realtime.h`](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h) | {ref}`Device handlers <using/realtime/host:Device Handler and Function ID>`, [host handlers](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/cudaq_realtime_host_api.md#host-dispatch-path-host_call), [device code](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/device_code_contract.md) |

### Dispatcher API

For applications: create a dispatcher, give it a function table with one entry
per handler (the `FNV-1a` hash of its name, its dispatch mode, the handler and
its argument schema), attach a bridge, then start and stop it.

### Bridge API

For transports: a plug-in library exports
`cudaq_realtime_get_bridge_interface`, which returns one table of function
pointers. Its version is checked when the library is loaded. `create`,
`connect`, `launch` and `disconnect` drive the transport; `get_ring_buffer`
hands over the rings, and `get_cpu_dataplane` or `get_gpu_dataplane` the
unified worker's hooks. Entries a transport does not support stay `NULL`.

### RPC protocol

Every RX slot holds a 24-byte `RPCHeader` followed by the arguments, packed in
schema order and little-endian. Every TX slot holds a 24-byte `RPCResponse`
followed by the results. Each bridge reports its slot size
(`cudaq_bridge_get_ring_geometry`); the CPU RoCE and `UDP` bridges default to
256 bytes.

| Bytes | Request: `RPCHeader` | Response: `RPCResponse` |
|-------|----------------------|-------------------------|
| 0 to 3 | `magic`, `'CUQR'` | `magic`, `'CUQS'` |
| 4 to 7 | `function_id`, the `FNV-1a` hash of the handler's name | `status`, 0 for success |
| 8 to 11 | `arg_len`, argument bytes | `result_len`, result bytes |
| 12 to 15 | `request_id`, chosen by the caller | `request_id`, echoed |
| 16 to 23 | `ptp_timestamp`, the send time | `ptp_timestamp`, echoed |
| 24 on | arguments | results |

### Handler calling convention

Every handler gets its request slot and its response slot. It reads the
arguments after the `RPCHeader`, writes its results after the `RPCResponse`,
and sets `result_len`. The two modes differ only in how the status comes back:

```cpp
// DEVICE_CALL, on the GPU: returns its status
__device__ int handler(const void *rx_slot, void *tx_slot, size_t slot_size);
// HOST_CALL, on the CPU: writes its status into tx_slot
void handler(const void *rx_slot, void *tx_slot, size_t slot_size);
```

Who writes each part of the response:

| Response field | `DEVICE_CALL` | `HOST_CALL` |
|----------------|---------------|-------------|
| result bytes | handler | handler |
| `result_len` | handler | handler |
| `status` | dispatcher, from the return value | handler |
| `magic`, `request_id`, `ptp_timestamp` | dispatcher | handler, copied from `rx_slot` |

Room for results is `slot_size - sizeof(RPCResponse)`. On an in-place ring
`tx_slot` is the same memory as `rx_slot`, so a handler reads what it needs from
the request before writing the response. A `DEVICE_CALL` handler runs on every
thread of the dispatcher's block; how it is built and handed to the dispatcher
is in [the device code page](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/device_code_contract.md).

## Three workers or one

The dispatcher and the transport can split the work two ways. Both hold on the
CPU path (threads) and the GPU path (persistent kernels).

### Three-worker model

Historically, the dispatcher met the bridge only through the rings. Three
independent loops spin: the bridge's RX worker, the dispatcher, and the
bridge's TX worker. None of them calls another; each tells the next that a slot
is ready by setting a per-slot flag in shared memory.

```{mermaid}
flowchart TB
    wireIn(["Wire in"])
    subgraph rxLoop [Bridge: RX worker loop]
        rxWorker["RX worker<br/>receive, write the slot, set rx_flags"]
    end
    rxRing[("RX ring<br/>slots and rx_flags")]
    subgraph dispatchLoop [Realtime: dispatch loop]
        dispatcher["Dispatcher<br/>poll rx_flags, run the handler, set tx_flags"]
    end
    txRing[("TX ring<br/>slots and tx_flags")]
    subgraph txLoop [Bridge: TX worker loop]
        txWorker["TX worker<br/>poll tx_flags, send the response"]
    end
    wireOut(["Wire out"])
    wireIn --> rxWorker --> rxRing --> dispatcher --> txRing --> txWorker --> wireOut
```

The rings and their flags are the whole contract: the bridge hands them over
with `get_ring_buffer` and the dispatcher polls them. Every request costs two
flag hand-offs between loops.

### Unified worker

The faster alternative runs one loop instead of three. The bridge also hands
the dispatcher two hooks into its transport, and the dispatcher calls them
itself: `rx_poll` claims the next ready request, `tx_publish` puts a response
on the wire.

```{mermaid}
sequenceDiagram
    participant W as Wire
    participant B as Bridge hooks
    participant D as Dispatcher, the only loop
    participant H as Handler
    W->>B: a request lands in slot n
    loop every turn
        D->>B: rx_poll(ctx, &slot)
        B-->>D: READY, slot n, or EMPTY
        D->>H: handler(rx_slot n, tx_slot n, slot_size)
        H-->>D: status
        D->>B: tx_publish(ctx, n)
    end
    B->>W: the response from slot n
```

It is almost the same API as the three-worker model: the same rings, the same
function table, the same handlers. What it adds is two hooks, which replace the
flag traffic between loops:

| | Three-worker model | Unified worker |
|--|--------------------|----------------|
| Loops | three: RX worker, dispatcher, TX worker | one: the dispatcher |
| Hand-off between them | per-slot flags in shared memory | function calls |
| What the bridge hands over | the rings, `get_ring_buffer` | the hooks: `get_cpu_dataplane`, with the rings, on the CPU path; `get_gpu_dataplane` on the GPU path |
| Dispatcher configuration | `CUDAQ_KERNEL_REGULAR` | `CUDAQ_KERNEL_UNIFIED` |

On the CPU path the hooks are plain function pointers in
`cudaq_cpu_dataplane_t`, next to the rings. On the GPU path they are device
functions (`cudaq_dev_enter_kernel`, `cudaq_dev_rx_poll`,
`cudaq_dev_tx_publish`) that the bridge ships as relocatable device code; the
dispatcher links them with its kernel and the application's handlers at run
time, when the application first looks up a handler.

The application does not wire the two models differently.
`cudaq_dispatcher_set_bridge` reads the configured `kernel_type` and asks the
bridge for the matching rings or hooks; it fails if the bridge was started for
the other model. All three in-tree bridges serve both:

| Bridge | Data path | Three-worker model | Unified worker |
|--------|-----------|--------------------|----------------|
| GPU RoCE | GPU | yes | GPU data plane |
| CPU RoCE | CPU | yes | CPU data plane |
| `UDP` | CPU | yes | CPU data plane |

## The GPU path in depth

Still to be written: customizing the dispatcher on the GPU path, how an
application provides its own dispatch kernel and handlers, and how the pieces
are linked into one CUDA module at run time. Until then, the
[device code page](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/device_code_contract.md)
covers building handlers and
[providing your own dispatch kernel](https://github.com/NVIDIA/cuda-quantum/blob/main/realtime/docs/device_code_contract.md#providing-your-own-dispatch-kernel).
