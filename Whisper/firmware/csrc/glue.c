/*
 * Platform glue that is impractical to write in stable Rust.
 *
 * nrf_axon_platform_printf is variadic; stable Rust cannot *define* variadic
 * functions (the c_variadic feature is unstable), so it lives here. Route it to
 * RTT/UART if you want the driver's diagnostic output; for now it is a sink.
 *
 * Everything else in the platform interface is implemented in src/platform.rs.
 */
#include <stdarg.h>

void nrf_axon_platform_printf(const char *fmt, ...)
{
	(void)fmt;
}
