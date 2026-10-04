#!/usr/bin/env python3
# Apple Silicon energy estimate. Requires Python 3, but no sudo.
# Counts modeled chip and DRAM energy, including other running processes.
# Excludes power supply losses, fans, and external devices.

import ctypes as c
import re
import signal
import subprocess
import sys
import time


def fail(message):
    print("joules: " + message, file=sys.stderr)
    sys.exit(1)

if sys.platform != "darwin":
    fail("requires macOS on Apple Silicon")

cf = c.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
io = c.CDLL("/usr/lib/libIOReport.dylib")
ptr = c.c_void_p

def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args
    return fn

string_new = bind(cf, "CFStringCreateWithCString", ptr, ptr, c.c_char_p, c.c_uint32)
string_get = bind(cf, "CFStringGetCString", c.c_bool, ptr, ptr, c.c_long, c.c_uint32)
dict_get = bind(cf, "CFDictionaryGetValue", ptr, ptr, ptr)
array_count = bind(cf, "CFArrayGetCount", c.c_long, ptr)
array_get = bind(cf, "CFArrayGetValueAtIndex", ptr, ptr, c.c_long)
release = bind(cf, "CFRelease", None, ptr)
copy_channels = bind(io, "IOReportCopyChannelsInGroup", ptr, ptr, ptr, c.c_uint64, c.c_uint64, c.c_uint64)
subscribe = bind(io, "IOReportCreateSubscription", ptr, ptr, ptr, c.POINTER(ptr), c.c_uint64, ptr)
sample = bind(io, "IOReportCreateSamples", ptr, ptr, ptr, ptr)
channel_name = bind(io, "IOReportChannelGetChannelName", ptr, ptr)
channel_unit = bind(io, "IOReportChannelGetUnitLabel", ptr, ptr)
channel_value = bind(io, "IOReportSimpleGetIntegerValue", c.c_int64, ptr, c.c_int)

def string(value):
    buf = c.create_string_buffer(1024)
    if not value or not string_get(value, buf, len(buf), 0x08000100):
        fail("cannot read IOReport string")
    return buf.value.decode("utf-8")

group = string_new(None, b"Energy Model", 0x08000100)
key = string_new(None, b"IOReportChannels", 0x08000100)
channels = copy_channels(group, None, 0, 0, 0)
if not channels:
    fail("Energy Model counters are unavailable")
subscribed = ptr()
subscription = subscribe(None, channels, c.byref(subscribed), 0, None)
if not subscription or not subscribed:
    fail("cannot subscribe to energy counters")

# Select CPU aggregate and individual SoC blocks, not their overlapping aliases.
blocks = re.compile(r"(?:DIE_\d+_)?(?:CPU Energy|GPU(?: SRAM)?\d*|ANE\d*|DRAM\d*|ISP\d*|AVE\d*|MSR\d*|AMCC\d*|DCS\d*|DISP\d*|DISPEXT\d*|VDEC\d*|SOC_AON\d*|SOC_REST\d*)")
units = {"J": 1, "mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}

def read():
    data = sample(subscription, subscribed, None)
    if not data:
        fail("cannot sample energy counters")
    values = {}
    items = dict_get(data, key)
    if not items:
        fail("sample has no channels")
    for i in range(array_count(items)):
        item = array_get(items, i)
        name = string(channel_name(item))
        if not blocks.fullmatch(name):
            continue
        unit = string(channel_unit(item))
        if unit not in units or name in values:
            fail("unexpected unit or duplicate channel: " + name)
        values[name] = channel_value(item, 0) * units[unit]
    release(data)
    if not any(name.endswith("CPU Energy") for name in values) or not any(re.search(r"(?:^|_)ANE\d*$", name) for name in values):
        fail("expected CPU/ANE counters are unavailable on this hardware")
    return values


def main():
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} command [args...]", file=sys.stderr)
        sys.exit(2)

    before = read()
    print("joules: modeled chip + DRAM energy; whole machine, not command-only or wall energy", file=sys.stderr)
    print("joules: channels: " + ", ".join(before), file=sys.stderr)
    print(before)
    print(f"joules: before = {sum(before.values()):.6f} J", file=sys.stderr, flush=True)
    start = time.monotonic()
    try:
        child = subprocess.Popen(sys.argv[1:])
    except OSError as e:
        print("joules: " + str(e), file=sys.stderr)
        sys.exit(127 if isinstance(e, FileNotFoundError) else 126)

    def forward(signum, frame):
        child.send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    status = child.wait()
    elapsed = time.monotonic() - start
    after = read()
    if after.keys() != before.keys() or any(after[k] < before[k] for k in before):
        print("joules: counters changed or reset; measurement invalid", file=sys.stderr)
    else:
        delta = sum(after[k] - before[k] for k in before)
        print(f"joules: after  = {sum(after.values()):.6f} J", file=sys.stderr)
        print(f"joules: consumed = {delta:.6f} J in {elapsed:.3f} s ({delta / elapsed:.3f} W average)", file=sys.stderr)
    sys.exit(status if status >= 0 else 128 - status)


if __name__ == "__main__":
    main()
