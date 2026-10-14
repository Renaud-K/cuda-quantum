# Device code for the unified GPU dispatcher

A DEVICE + UNIFIED dispatcher (`CUDAQ_DISPATCH_PATH_DEVICE`,
`CUDAQ_KERNEL_UNIFIED`) runs one persistent kernel that receives a request,
calls a DEVICE_CALL handler and sends the response.  That kernel needs code
from three parties, and CUDA only resolves device calls inside one module:

| Image | Built by | Handed over through |
|-------|----------|---------------------|
| Dispatcher: `cudaq_unified_dispatch_kernel` | CUDA-Q realtime, embedded in `libcudaq-realtime.so` | (internal) |
| Transport: the three hooks of `unified_device_transport.cuh` | the bridge provider's repository | `get_gpu_dataplane` (`cudaq_gpu_dataplane_t`) |
| Handlers: the application's DEVICE_CALL functions | the application's repository | `cudaq_dispatcher_set_device_code` |

`cudaq_dispatcher_set_device_code` links the three with nvJitLink into one
module for the GPU in use, with device link-time optimization, and loads it.
The transport hooks are then inlined into the kernel; the handlers stay
indirect calls through the function table.

The three images usually come from different repositories and build
systems.  This document is what realtime requires of the two it does not
build.  `cudaq_realtime_add_device_code` (installed with the
`cudaq-realtime` CMake package) implements it for CMake projects, but any
build that follows it works.

## Recipe

```sh
nvcc -fatbin -rdc=true \
     -gencode arch=compute_90,code=[sm_90,lto_90] \
     -gencode arch=compute_100,code=[sm_100,lto_100] \
     -I<cudaq-realtime>/include my_handlers.cu -o my_handlers.fatbin
```

Embed the file in your library or executable with any tool (`bin2c`,
`xxd -i`, `.incbin`, `#embed`) and describe it as a `cudaq_device_image_t`
(`device_code.h`): a pointer and a size.  Several translation units become
several images; both hand-over points take an array.

With CMake:

```cmake
find_package(cudaq-realtime REQUIRED)
cudaq_realtime_add_device_code(my_app
  SYMBOL my_handler_images
  SOURCES my_handlers.cu)
# my_app can now #include "my_handler_images.h", which declares
#   extern "C" const cudaq_device_image_t my_handler_images[];
#   extern "C" const size_t my_handler_images_count;
```

## Requirements on both images

- **Relocatable, never device-linked.** Compile with `-rdc=true` and keep the
  result of compilation (`-fatbin`).  The output of a device link (`-dlink`,
  or the image inside an executable built with `CUDA_RESOLVE_DEVICE_SYMBOLS`)
  is final and cannot be linked again.
- **Code for the GPU it runs on.** Each image needs, for the running
  architecture, LTO-IR (`code=lto_XX`, preferred: it is what allows the
  hooks to be inlined), relocatable SASS (`code=sm_XX`), or PTX as a slower
  fallback compiled at load time.  LTO-IR for an older architecture also
  works (`lto_90` linked for sm_100).  Mixing forms across images is fine.
- **A toolkit no newer than realtime's nvJitLink.** nvJitLink rejects any
  input built by a newer CUDA toolkit than itself, in every form: LTO-IR
  ("may need newer version of nvJitLink library"), SASS (a bare "bad input")
  and PTX (an unsupported `.version`).  Inputs from an older toolkit of the
  same major version link.  Building with the same CUDA major version as
  realtime, at or below its minor version, is safe.  Mixing CUDA major
  versions is not tested.
- **Self-contained.** Everything the image calls must be in it: header-only
  device code, or sources compiled into the same fatbin.  A separate device
  library (`.a`) cannot be passed alongside it.
- **No device runtime.** The link does not include `cudadevrt`, so no
  dynamic parallelism (`<<<>>>` in device code) and no device-side
  `cudaGraphLaunch`.  Device `printf` works.
- **Private symbols.** Only the names this contract defines are shared
  between images.  Every other non-inline device function or variable must be
  `static` or in an anonymous namespace: a strong symbol defined in two
  images fails the link ("symbol multiply defined").  Templates and `inline`
  functions are fine.  The `cudaq_` prefix is reserved for realtime.
- **No state set from the host in your own module.** Module globals are
  duplicated into the linked module, so `cudaMemcpyToSymbol` in your own
  module writes a copy the kernel never reads.  Reach state through
  pointers, or initialize it in the linked module through
  `cudaq_dispatcher_get_device_global`.
- **Valid while realtime reads it.** Realtime reads an image only during
  `cudaq_dispatcher_set_device_code`.

## The transport image (bridge provider)

- Defines exactly `cudaq_dev_transport_attach`, `cudaq_dev_rx_poll` and
  `cudaq_dev_tx_publish` with the signatures and contracts in
  `unified_device_transport.cuh`, built against the realtime headers whose
  `CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION` the provider reports.
- Fits its session in `CUDAQ_DEV_TRANSPORT_SESSION_BYTES` and
  `CUDAQ_DEV_TRANSPORT_SESSION_ALIGN` (check with a `static_assert`).  Any
  `__shared__` state it declares comes out of the kernel's single block.
- Is returned from `get_gpu_dataplane` together with `ctx`, the
  device-accessible state handed to `attach`.  Both stay valid while the
  bridge exists.

## The handler image (application)

- Each handler has the `DeviceRPCFunction` signature
  (`dispatch_kernel_launch.h`) and is exported, in the translation unit that
  defines it, with `CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(handler)`
  (`device_code.h`).  Exported names must be unique across the images.
- The image contains the handlers and what they call, nothing else: not the
  rest of the application's device code, not `libcudaq-realtime-dispatch.a`,
  not a transport.
- The application still builds the function table, but from the addresses
  `cudaq_dispatcher_get_device_function` returns, never from `&handler` taken
  in its own module.  `cudaq_dispatcher_start` refuses any DEVICE_CALL entry
  whose pointer did not come from there.

## Wiring sequence

```c
cudaq_dispatcher_create(manager, &config, &dispatcher);  // DEVICE + UNIFIED
cudaq_dispatcher_set_bridge(dispatcher, bridge);         // transport image
cudaq_dispatcher_set_device_code(dispatcher, my_handler_images,
                                 my_handler_images_count); // links
void *fn;
cudaq_dispatcher_get_device_function(dispatcher, "my_handler", &fn);
/* fill entries[i].handler.device_fn_ptr = fn, copy the table to the device */
cudaq_dispatcher_set_function_table(dispatcher, &table);
cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats);
cudaq_dispatcher_start(dispatcher);
```

A provider that replaces the dispatch loop with its own kernel (a launch
override from `get_transport_context(UNIFIED)`) links nothing at runtime;
its kernel must share a module with the handlers by other means, and
`cudaq_dispatcher_set_device_code` returns `CUDAQ_ERR_UNSUPPORTED` for it.
