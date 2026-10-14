/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_gpu_dataplane.cpp
/// @brief The DEVICE + UNIFIED dispatcher over a provider's GPU data plane.
///
/// Three device images built apart -- the dispatcher's kernel (inside
/// libcudaq-realtime.so), the loopback provider's hooks (inside its .so) and
/// this test's handler (embedded here) -- are linked at runtime by
/// cudaq_dispatcher_set_device_code.  The test plays the wire through the
/// provider's mapped rings, so it needs a GPU but no NIC.

#include "loopback_handler_images.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

#ifndef GPU_LOOPBACK_PROVIDER_PATH
#error "GPU_LOOPBACK_PROVIDER_PATH must name the built loopback provider .so"
#endif

namespace {

using cudaq::realtime::RPCResponse;

constexpr std::uint32_t kAddId = cudaq::realtime::fnv1a_hash("loopback_add");

class GpuDataplaneTest : public ::testing::Test {
protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
      GTEST_SKIP() << "no CUDA device";
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager), CUDAQ_OK);
    ASSERT_EQ(cudaq_bridge_create_from_library(
                  &bridge, GPU_LOOPBACK_PROVIDER_PATH, 0, nullptr),
              CUDAQ_OK);
    ASSERT_EQ(cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &ring),
              CUDAQ_OK);

    void *flag = nullptr;
    void *d_flag = nullptr;
    ASSERT_EQ(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped),
              cudaSuccess);
    ASSERT_EQ(cudaHostGetDevicePointer(&d_flag, flag, 0), cudaSuccess);
    shutdown_flag = static_cast<volatile int *>(flag);
    d_shutdown_flag = static_cast<volatile int *>(d_flag);
    *shutdown_flag = 0;
    ASSERT_EQ(cudaMalloc(&d_stats, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_stats, 0, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_entries, sizeof(cudaq_function_entry_t)),
              cudaSuccess);
  }

  void TearDown() override {
    if (dispatcher) {
      // A running kernel waits for this flag; stop() alone would queue behind
      // it.
      *shutdown_flag = 1;
      cudaq_dispatcher_stop(dispatcher);
      cudaq_dispatcher_destroy(dispatcher);
    }
    if (bridge)
      EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    cudaFree(d_entries);
    cudaFree(d_stats);
    if (shutdown_flag)
      cudaFreeHost(const_cast<int *>(shutdown_flag));
  }

  cudaq_dispatcher_t *
  make_dispatcher(cudaq_dispatch_path_t path = CUDAQ_DISPATCH_PATH_DEVICE) {
    cudaq_dispatcher_config_t config{};
    config.dispatch_path = path;
    config.kernel_type = CUDAQ_KERNEL_UNIFIED;
    config.dispatch_mode = path == CUDAQ_DISPATCH_PATH_DEVICE
                               ? CUDAQ_DISPATCH_DEVICE_CALL
                               : CUDAQ_DISPATCH_HOST_CALL;
    config.num_blocks = 1;
    config.threads_per_block = 1;
    EXPECT_EQ(cudaq_dispatcher_create(manager, &config, &dispatcher), CUDAQ_OK);
    return dispatcher;
  }

  cudaq_status_t link() {
    return cudaq_dispatcher_set_device_code(dispatcher, loopback_handler_images,
                                            loopback_handler_images_count);
  }

  // The application's half of the table: one entry, pointing wherever the
  // caller says.
  void set_table(void *handler) {
    cudaq_function_entry_t entry{};
    entry.handler.device_fn_ptr = handler;
    entry.function_id = kAddId;
    entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    ASSERT_EQ(
        cudaMemcpy(d_entries, &entry, sizeof(entry), cudaMemcpyHostToDevice),
        cudaSuccess);
    cudaq_function_table_t table{d_entries, 1};
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher, &table),
              CUDAQ_OK);
    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats),
        CUDAQ_OK);
  }

  // Play the wire for one request: write it, raise its RX flag, and wait for
  // the transport's TX flag.
  bool round_trip(std::uint32_t slot, std::uint32_t request_id,
                  const std::vector<std::uint8_t> &payload,
                  std::vector<std::uint8_t> &result, RPCResponse &response) {
    if (cudaq_host_ringbuffer_write_rpc_request(
            &ring, slot, kAddId, payload.data(),
            static_cast<std::uint32_t>(payload.size()), request_id,
            0) != CUDAQ_OK)
      return false;
    cudaq_host_ringbuffer_signal_slot(&ring, slot);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (cudaq_host_ringbuffer_poll_tx_flag(&ring, slot, nullptr) !=
           CUDAQ_TX_READY) {
      if (std::chrono::steady_clock::now() > deadline)
        return false;
      std::this_thread::yield();
    }
    const std::uint8_t *frame = ring.tx_data_host + slot * ring.tx_stride_sz;
    std::memcpy(&response, frame, sizeof(response));
    result.assign(frame + sizeof(response),
                  frame + sizeof(response) + response.result_len);
    cudaq_host_ringbuffer_clear_slot(&ring, slot);
    return true;
  }

  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;
  cudaq_ringbuffer_t ring{};
  volatile int *shutdown_flag = nullptr;
  volatile int *d_shutdown_flag = nullptr;
  std::uint64_t *d_stats = nullptr;
  cudaq_function_entry_t *d_entries = nullptr;
};

// The handler, its module state and the transport hooks all come from images
// the dispatcher linked; requests wrap the ring to reuse every slot.
TEST_F(GpuDataplaneTest, ServesRequestsThroughTheLinkedModule) {
  make_dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(dispatcher, bridge), CUDAQ_OK);
  ASSERT_EQ(link(), CUDAQ_OK);

  void *increment = nullptr;
  size_t size = 0;
  ASSERT_EQ(cudaq_dispatcher_get_device_global(dispatcher, "loopback_increment",
                                               &increment, &size),
            CUDAQ_OK);
  ASSERT_EQ(size, sizeof(unsigned));
  const unsigned two = 2;
  ASSERT_EQ(cudaMemcpy(increment, &two, sizeof(two), cudaMemcpyHostToDevice),
            cudaSuccess);

  void *handler = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_function(dispatcher, "loopback_add",
                                                 &handler),
            CUDAQ_OK);
  ASSERT_NE(handler, nullptr);
  set_table(handler);
  ASSERT_EQ(cudaq_dispatcher_start(dispatcher), CUDAQ_OK);

  uint32_t num_slots = 0;
  uint32_t slot_size = 0;
  ASSERT_EQ(cudaq_bridge_get_ring_geometry(bridge, &num_slots, &slot_size),
            CUDAQ_OK);
  const std::uint32_t requests = num_slots + num_slots / 2;
  for (std::uint32_t i = 0; i < requests; ++i) {
    const std::vector<std::uint8_t> payload = {static_cast<std::uint8_t>(i), 10,
                                               20, 250};
    std::vector<std::uint8_t> result;
    RPCResponse response{};
    ASSERT_TRUE(round_trip(i % num_slots, 100 + i, payload, result, response))
        << "request " << i << " got no response";
    EXPECT_EQ(response.magic, CUDAQ_RPC_MAGIC_RESPONSE);
    EXPECT_EQ(response.status, 0);
    EXPECT_EQ(response.request_id, 100 + i);
    ASSERT_EQ(result.size(), payload.size());
    for (size_t b = 0; b < payload.size(); ++b)
      EXPECT_EQ(result[b], static_cast<std::uint8_t>(payload[b] + 2));
  }

  *shutdown_flag = 1;
  ASSERT_EQ(cudaq_dispatcher_stop(dispatcher), CUDAQ_OK);
  std::uint64_t processed = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(dispatcher, &processed), CUDAQ_OK);
  EXPECT_EQ(processed, requests);
}

// An address that did not come out of the linked module would fault on the
// first request, so start() refuses it up front.
TEST_F(GpuDataplaneTest, RefusesAHandlerFromOutsideTheLinkedModule) {
  make_dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(dispatcher, bridge), CUDAQ_OK);
  ASSERT_EQ(link(), CUDAQ_OK);
  void *not_a_handler = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_global(dispatcher, "loopback_increment",
                                               &not_a_handler, nullptr),
            CUDAQ_OK);
  set_table(not_a_handler);
  EXPECT_EQ(cudaq_dispatcher_start(dispatcher), CUDAQ_ERR_INVALID_ARG);
}

// The transport's device code comes from the bridge, so the link cannot
// happen before set_bridge, and start() cannot happen before the link.
TEST_F(GpuDataplaneTest, LinksOnlyAfterTheBridgeAndBeforeStart) {
  make_dispatcher();
  EXPECT_EQ(link(), CUDAQ_ERR_INVALID_ARG);
  ASSERT_EQ(cudaq_dispatcher_set_bridge(dispatcher, bridge), CUDAQ_OK);
  set_table(reinterpret_cast<void *>(0x1));
  EXPECT_EQ(cudaq_dispatcher_start(dispatcher), CUDAQ_ERR_INVALID_ARG);
  EXPECT_EQ(link(), CUDAQ_OK);
  EXPECT_EQ(link(), CUDAQ_ERR_INVALID_ARG);
}

// A failed link leaves the dispatcher unlinked, so a good image still links.
TEST_F(GpuDataplaneTest, ReportsALinkFailureAndCanLinkAgain) {
  make_dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(dispatcher, bridge), CUDAQ_OK);
  static const char garbage[] = "not a fatbin";
  const cudaq_device_image_t bad = {garbage, sizeof(garbage)};
  EXPECT_EQ(cudaq_dispatcher_set_device_code(dispatcher, &bad, 1),
            CUDAQ_ERR_INVALID_ARG);
  void *fn = nullptr;
  EXPECT_EQ(
      cudaq_dispatcher_get_device_function(dispatcher, "loopback_add", &fn),
      CUDAQ_ERR_INVALID_ARG);
  EXPECT_EQ(link(), CUDAQ_OK);
  EXPECT_EQ(
      cudaq_dispatcher_get_device_function(dispatcher, "loopback_add", &fn),
      CUDAQ_OK);
}

TEST_F(GpuDataplaneTest, RefusesNamesTheModuleDoesNotHave) {
  make_dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(dispatcher, bridge), CUDAQ_OK);
  ASSERT_EQ(link(), CUDAQ_OK);
  void *fn = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_function(dispatcher, "missing", &fn),
            CUDAQ_ERR_INVALID_ARG);
  void *ptr = nullptr;
  EXPECT_EQ(
      cudaq_dispatcher_get_device_global(dispatcher, "missing", &ptr, nullptr),
      CUDAQ_ERR_INVALID_ARG);
}

TEST_F(GpuDataplaneTest, OnlyADeviceUnifiedDispatcherLinksDeviceCode) {
  make_dispatcher(CUDAQ_DISPATCH_PATH_HOST);
  EXPECT_EQ(link(), CUDAQ_ERR_INVALID_ARG);
}

} // namespace
