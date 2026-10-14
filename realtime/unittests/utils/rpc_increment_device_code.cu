/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file rpc_increment_device_code.cu
/// @brief Handler image for the DEVICE + UNIFIED dispatcher: the increment
/// handler, exported so cudaq_dispatcher_get_device_function finds it in the
/// module cudaq_dispatcher_set_device_code links.

#include "rpc_increment_handler.cuh"
#include "cudaq/realtime/daemon/dispatcher/device_code.h"

CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(rpc_increment_handler)
