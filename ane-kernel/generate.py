"""Compile fused planar PQ2_0 decode/matmul programs for the private ANE runtime."""
import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models")
parser.add_argument("--rows", type=int, nargs="+", default=[11264, 6656, 3840, 7936])
parser.add_argument("--widths", type=int, nargs="+", default=[5120])
parser.add_argument("--tokens", type=int, nargs="+", default=[2048])
parser.add_argument("--chunk-k", type=int, default=2048)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
assert args.chunk_k > 0 and args.chunk_k % 128 == 0


def decode(q, scale):
    digits = [q] + [mb.floor(x=mb.real_div(x=q, y=np.float16(4**i))) for i in range(1, 4)]
    fields = [mb.sub(x=mb.sub(x=digits[i], y=mb.mul(x=digits[i+1], y=np.float16(4))), y=np.float16(1)) for i in range(3)]
    fields.append(mb.sub(x=digits[3], y=np.float16(1)))
    return mb.mul(x=mb.concat(values=fields, axis=3, interleave=False), y=scale)


for k in args.widths:
    assert k > 0 and k % 128 == 0
    blocks = k // 128
    for rows in args.rows:
        for tokens in args.tokens:
            name = f"pq2_k{k}_r{rows}_t{tokens}"

            @mb.program(input_specs=[
                mb.TensorSpec(shape=(1, 1, rows, k//4), dtype=types.uint8),
                mb.TensorSpec(shape=(1, 1, rows, (blocks+31)//32*32), dtype=types.fp16),
                mb.TensorSpec(shape=(1, tokens, blocks, 128) if k > 16384 else (1, 1, tokens, k), dtype=types.fp16),
            ], opset_version=ct.target.iOS18)
            def program(q, d, act):
                d = mb.slice_by_size(x=d, begin=[0, 0, 0, 0], size=[1, 1, rows, blocks])
                d = mb.reshape(x=d, shape=[1, rows, blocks, 1])
                d = mb.mul(x=d, y=np.float16(32))
                q = mb.reshape(x=mb.cast(x=q, dtype="fp16"), shape=[1, rows, blocks, 32])
                act = mb.cast(x=act, dtype="fp16")
                if k > 16384:
                    act = mb.reshape(x=act, shape=[1, 1, tokens, k])
                result = None
                for first in range(0, blocks, args.chunk_k//128):
                    count = min(args.chunk_k//128, blocks-first)
                    qp = mb.slice_by_size(x=q, begin=[0, 0, first, 0], size=[1, rows, count, 32])
                    dp = mb.slice_by_size(x=d, begin=[0, 0, first, 0], size=[1, rows, count, 1])
                    weights = mb.reshape(x=decode(qp, dp), shape=[1, 1, rows, count*128])
                    ap = mb.slice_by_size(x=act, begin=[0, 0, 0, first*128], size=[1, 1, tokens, count*128])
                    ap = mb.reshape(x=ap, shape=[1, 1, tokens, count*128])
                    partial = mb.matmul(x=ap, y=weights, transpose_y=True)
                    result = partial if result is None else mb.add(x=result, y=partial)
                return mb.cast(x=result, dtype="fp16", name="result")

            model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                               inputs=[ct.ImageType(name="q", color_layout=ct.colorlayout.GRAYSCALE, grayscale_use_uint8=True)],
                               pass_pipeline=ct.PassPipeline.EMPTY, skip_model_load=True)
            model.user_defined_metadata.update(pq2_output_scale="32", pq2_layout="block-planes", pq2_private_ane="True")
            package = args.output / (name + ".mlpackage")
            model.save(str(package))
            runtime = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
            compiled = runtime.get_compiled_model_path()
            plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
            devices = []
            for op in plan.model_structure.program.functions["main"].block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage:
                    devices.append({"op": op.operator_name, "device": type(usage.preferred_compute_device).__name__})
            if not devices or any(x["device"] != "MLNeuralEngineComputeDevice" for x in devices):
                raise RuntimeError(f"{name}: not all operations prefer ANE: {devices}")
            (args.output / (name + ".plan.json")).write_text(json.dumps(devices, indent=2) + "\n")
            destination = args.output / (name + ".mlmodelc")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(compiled, destination)
            print(name, "all reported operations prefer ANE", flush=True)
