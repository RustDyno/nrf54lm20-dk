/* bindgen entry point: the public Axon API surface we call from Rust. */
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_platform_interface.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "drivers/axon/nrf_axon_nn_op_extensions.h"

/* When you add a compiled model, also bind its generated header here, e.g.:
 *   #include "generated/nrf_axon_model_<name>_.h"
 * and add its include dir to the bindgen + cc include paths in build.rs.
 */
