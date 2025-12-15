import functools
import numbers

from collections.abc import Iterable
from itertools import product

import numpy as np
from taichi._lib import core as ti_python_core
from taichi._lib.utils import ti_python_core as _ti_python_core
from taichi.lang import expr, impl, ops as ops_mod, runtime_ops
from taichi.lang._ndarray import Ndarray, NdarrayHostAccess
from taichi.lang.common_ops import TaichiOperations
from taichi.lang.enums import Layout
from taichi.lang.exception import (
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
)
from taichi.lang.field import Field, ScalarField, SNodeHostAccess
from taichi.lang.util import (
    cook_dtype,
    get_traceback,
    in_python_scope,
    python_scope,
    taichi_scope,
    to_numpy_type,
    to_paddle_type,
    to_pytorch_type,
    warning,
)
from taichi.types import primitive_types
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed

_type_factory = _ti_python_core.get_type_factory_instance()


def _generate_swizzle_patterns(key_group: str, required_length=4):
    result = []
    for k in range(1, required_length + 1):
        result.extend(product(key_group, repeat=k))
    return ["".join(pat) for pat in result]


def _gen_swizzles(cls):
    KEYGROUP_SET = ["xyzw", "rgba", "stpq"]
    cls._swizzle_to_keygroup = {}
    cls._keygroup_to_checker = {}

    def make_checker(key_group):
        def check(instance, pattern):
            valid = set(key_group[: instance.n])
            diff = set(pattern) - valid
            if diff:
                raise TaichiSyntaxError(
                    f"vec{instance.n} only has {tuple(sorted(valid))}, got={tuple(pattern)}"
                )

        return check

    for key_group in KEYGROUP_SET:
        cls._keygroup_to_checker[key_group] = make_checker(key_group)
        for idx, attr in enumerate(key_group):

            def gen_prop(attr, attr_idx, kg):
                checker = cls._keygroup_to_checker[kg]

                def getter(inst):
                    checker(inst, attr)
                    return inst[attr_idx]

                @python_scope
                def setter(inst, val):
                    checker(inst, attr)
                    inst[attr_idx] = val

                return property(getter, setter)

            setattr(cls, attr, gen_prop(attr, idx, key_group))
            cls._swizzle_to_keygroup[attr] = key_group

    for key_group in KEYGROUP_SET:
        patterns = filter(lambda p: len(p) > 1, _generate_swizzle_patterns(key_group, 4))
        for pat in patterns:

            def gen_prop(pattern, kg):
                checker = cls._keygroup_to_checker[kg]

                def getter(inst):
                    checker(inst, pattern)
                    return Vector([inst[kg.index(ch)] for ch in pattern])

                @python_scope
                def setter(inst, val):
                    if len(pattern) != len(val):
                        raise TaichiRuntimeError(f"len mismatch for {pattern}")
                    checker(inst, pattern)
                    for ch, v in zip(pattern, val):
                        inst[kg.index(ch)] = v

                return property(getter, setter)

            setattr(cls, pat, gen_prop(pat, key_group))
            cls._swizzle_to_keygroup[pat] = key_group
    return cls


def _infer_entry_dt(entry):
    if isinstance(entry, (int, np.integer)):
        return impl.get_runtime().default_ip
    if isinstance(entry, (float, np.floating)):
        return impl.get_runtime().default_fp
    if isinstance(entry, expr.Expr):
        dt = entry.ptr.get_rvalue_type()
        if dt == ti_python_core.DataType_unknown:
            raise TaichiTypeError("Cannot infer element type")
        return dt
    raise TaichiTypeError("Invalid element type")


def _infer_array_dt(arr):
    return functools.reduce(ti_python_core.promoted_type, map(_infer_entry_dt, arr))


def make_matrix_with_shape(arr, shape, dt):
    return expr.Expr(
        impl.get_runtime()
        .compiling_callable.ast_builder()
        .make_matrix_expr(
            shape,
            dt,
            [expr.Expr(elt).ptr for elt in arr],
            ti_python_core.DebugInfo(impl.get_runtime().get_current_src_info()),
        )
    )


def make_matrix(arr, dt=None):
    if not arr:
        shape, dt = [0], primitive_types.i32
    else:
        shape = [len(arr), len(arr[0])] if isinstance(arr[0], Iterable) else [len(arr)]
        if isinstance(arr[0], Iterable):
            arr = [e for row in arr for e in row]
        dt = _infer_array_dt(arr) if dt is None else cook_dtype(dt)
    return make_matrix_with_shape(arr, shape, dt)


def _read_host_access(x):
    return x.accessor.getter(*x.key) if isinstance(x, SNodeHostAccess) else x.getter()


def _write_host_access(x, val):
    (x.accessor.setter(val, *x.key) if isinstance(x, SNodeHostAccess) else x.setter(val))


@_gen_swizzles
class Matrix(TaichiOperations):
    _is_taichi_class = _is_matrix_class = True
    __array_priority__ = 1000

    def __init__(self, arr, dt=None):
        if not isinstance(arr, (list, tuple, np.ndarray)):
            raise TaichiTypeError("Matrix needs array-like input")

        if not arr:
            self.ndim = self.n = self.m = 0
            self.entries, self.is_host_access = np.array([]), False
        elif isinstance(arr[0], Matrix):
            raise Exception("cols/rows required with vector list")
        elif isinstance(arr[0], Iterable):
            self.ndim, self.n, self.m = 2, len(arr), len(arr[0])
            self.is_host_access = isinstance(arr[0][0], (SNodeHostAccess, NdarrayHostAccess))
            self.entries = arr if self.is_host_access else np.array(arr, None if dt is None else to_numpy_type(dt))
        else:
            self.ndim, self.n, self.m = 1, len(arr), 1
            self.is_host_access = isinstance(arr[0], (SNodeHostAccess, NdarrayHostAccess))
            self.entries = arr if self.is_host_access else np.array(arr, None if dt is None else to_numpy_type(dt))

        if self.n * self.m > 32:
            warning(
                f"Matrix {self.n}x{self.m} > 32 may slow compilation. Use field for large matrices.",
                UserWarning,
                stacklevel=2,
            )

    def get_shape(self):
        return (self.n,) if self.ndim == 1 else (self.n, self.m) if self.ndim == 2 else None

    def __matmul__(self, other):
        from taichi.lang import matrix_ops
        return matrix_ops.matmul(self, other)

    def __len__(self):
        return self.n

    def __iter__(self):
        return (
            (self[i] for i in range(self.n))
            if self.ndim == 1
            else ([self[i, j] for j in range(self.m)] for i in range(self.n))
        )

    def __getitem__(self, indices):
        entry = self._get_entry(indices)
        return _read_host_access(entry) if self.is_host_access else entry

    @python_scope
    def __setitem__(self, indices, item):
        if self.is_host_access:
            _write_host_access(self._get_entry(indices), item)
        else:
            indices = [indices] if not isinstance(indices, (list, tuple)) else indices
            assert len(indices) == self.ndim
            self.entries[indices[0]] = item if self.ndim == 1 else None
            if self.ndim == 2:
                self.entries[indices[0]][indices[1]] = item

    def _get_entry(self, indices):
        indices = [indices] if not isinstance(indices, (list, tuple)) else indices
        assert len(indices) == self.ndim
        return self.entries[indices[0]] if self.ndim == 1 else self.entries[indices[0]][indices[1]]

    @python_scope
    def _set_entries(self, value):
        value = value.to_list() if isinstance(value, Matrix) else value
        if self.is_host_access:
            if self.ndim == 1:
                for i in range(self.n):
                    _write_host_access(self.entries[i], value[i])
            else:
                for i in range(self.n):
                    for j in range(self.m):
                        _write_host_access(self.entries[i][j], value[i][j])
        else:
            if self.ndim == 1:
                for i in range(self.n):
                    self.entries[i] = value[i]
            else:
                for i in range(self.n):
                    for j in range(self.m):
                        self.entries[i][j] = value[i][j]

    @property
    def _members(self):
        return self.entries

    def to_list(self):
        if self.is_host_access:
            return (
                [_read_host_access(self.entries[i]) for i in range(self.n)]
                if self.ndim == 1
                else [[_read_host_access(self.entries[i][j]) for j in range(self.m)] for i in range(self.n)]
            )
        return self.entries.tolist()

    @taichi_scope
    def cast(self, dtype):
        return (
            Vector([ops_mod.cast(self[i], dtype) for i in range(self.n)])
            if self.ndim == 1
            else Matrix([[ops_mod.cast(self[i, j], dtype) for j in range(self.m)] for i in range(self.n)])
        )

    def trace(self):
        from taichi.lang import matrix_ops
        return matrix_ops.trace(self)

    def inverse(self):
        from taichi.lang import matrix_ops
        return matrix_ops.inverse(self)

    def normalized(self, eps=0):
        from taichi.lang import matrix_ops
        return matrix_ops.normalized(self, eps)

    def transpose(self):
        from taichi.lang import matrix_ops
        return matrix_ops.transpose(self)

    @taichi_scope
    def determinant(self):
        from taichi.lang import matrix_ops
        return matrix_ops.determinant(self)

    def sum(self):
        from taichi.lang import matrix_ops
        return matrix_ops.sum(self)

    def norm(self, eps=0):
        from taichi.lang import matrix_ops
        return matrix_ops.norm(self, eps=eps)

    def norm_inv(self, eps=0):
        from taichi.lang import matrix_ops
        return matrix_ops.norm_inv(self, eps=eps)

    def norm_sqr(self):
        from taichi.lang import matrix_ops
        return matrix_ops.norm_sqr(self)

    def max(self):
        from taichi.lang import matrix_ops
        return matrix_ops.max(self)

    def min(self):
        from taichi.lang import matrix_ops
        return matrix_ops.min(self)

    def any(self):
        from taichi.lang import matrix_ops
        return matrix_ops.any(self)

    def all(self):
        from taichi.lang import matrix_ops
        return matrix_ops.all(self)

    def fill(self, val):
        from taichi.lang import matrix_ops
        return matrix_ops.fill(self, val)

    def to_numpy(self):
        return np.array(self.to_list()) if self.is_host_access else self.entries

    @taichi_scope
    def __ti_repr__(self):
        yield "["
        for i in range(self.n):
            if i:
                yield ", "
            if self.m != 1:
                yield "["
            for j in range(self.m):
                if j:
                    yield ", "
                yield self(i, j)
            if self.m != 1:
                yield "]"
        yield "]"

    def __str__(self):
        return f"<{self.n}x{self.m} ti.Matrix>" if impl.inside_kernel() else str(self.to_numpy())

    def __repr__(self):
        return str(self.to_numpy())

    @staticmethod
    @taichi_scope
    def zero(dt, n, m=None):
        from taichi.lang import matrix_ops
        return matrix_ops._filled_vector(n, dt, 0) if m is None else matrix_ops._filled_matrix(n, m, dt, 0)

    @staticmethod
    @taichi_scope
    def one(dt, n, m=None):
        from taichi.lang import matrix_ops
        return matrix_ops._filled_vector(n, dt, 1) if m is None else matrix_ops._filled_matrix(n, m, dt, 1)

    @staticmethod
    @taichi_scope
    def unit(n, i, dt=None):
        from taichi.lang import matrix_ops
        assert 0 <= i < n
        return matrix_ops._unit_vector(n, i, dt or int)

    @staticmethod
    @taichi_scope
    def identity(dt, n):
        from taichi.lang import matrix_ops
        return matrix_ops._identity_matrix(n, dt)

    @staticmethod
    @taichi_scope
    def diag(dim, val):
        from taichi.lang import matrix_ops
        return matrix_ops.diag(dim, val)

    @classmethod
    @python_scope
    def field(
        cls,
        n,
        m,
        dtype,
        shape=None,
        order=None,
        name="",
        offset=None,
        needs_grad=False,
        needs_dual=False,
        layout=Layout.AOS,
        ndim=None,
    ):
        entries, element_dim = [], ndim if ndim is not None else 2

        if isinstance(dtype, (list, tuple, np.ndarray)):
            if m == 1:
                assert len(np.shape(dtype)) == 1 and len(dtype) == n
                entries = [
                    impl.create_field_member(dtype[i], name=name, needs_grad=needs_grad, needs_dual=needs_dual)
                    for i in range(n)
                ]
            else:
                assert len(np.shape(dtype)) == 2 and len(dtype) == n and len(dtype[0]) == m
                entries = [
                    impl.create_field_member(dtype[i][j], name=name, needs_grad=needs_grad, needs_dual=needs_dual)
                    for i in range(n)
                    for j in range(m)
                ]
        else:
            entries = [
                impl.create_field_member(dtype, name=name, needs_grad=needs_grad, needs_dual=needs_dual)
                for _ in range(n * m)
            ]

        entries, entries_grad, entries_dual = zip(*entries)
        entries = MatrixField(entries, n, m, element_dim)
        if all(entries_grad):
            entries._set_grad(MatrixField(entries_grad, n, m, element_dim))
        if all(entries_dual):
            entries._set_dual(MatrixField(entries_dual, n, m, element_dim))
        impl.get_runtime().matrix_fields.append(entries)

        # FIX: allow scalar fields with shape=() (since () is falsy, `if shape:` breaks it)
        if shape is None:
            if offset is not None:
                raise TaichiSyntaxError("shape cannot be None when offset is set")
            if order is not None:
                raise TaichiSyntaxError("shape cannot be None when order is set")
            return entries

        shape = (shape,) if isinstance(shape, numbers.Number) else shape
        offset = (offset,) if isinstance(offset, numbers.Number) else offset
        dim = len(shape)
        if offset is not None and dim != len(offset):
            raise TaichiSyntaxError(f"shape/offset dim mismatch: {dim} != {len(offset)}")

        axis_seq, shape_seq = list(range(dim)), list(shape)
        if order is not None:
            if dim != len(order) or dim != len(set(order)):
                raise TaichiSyntaxError("Invalid order")
            axis_seq = [ord(ch) - ord("i") for ch in order]
            shape_seq = [shape[ax] for ax in axis_seq]

        same_level = order is None
        if layout == Layout.SOA:
            for e in entries._get_field_members():
                impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
            if needs_grad:
                for e in entries_grad._get_field_members():
                    impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
            if needs_dual:
                for e in entries_dual._get_field_members():
                    impl._create_snode(axis_seq, shape_seq, same_level).place(ScalarField(e), offset=offset)
        else:
            impl._create_snode(axis_seq, shape_seq, same_level).place(entries, offset=offset)
            if needs_grad:
                impl._create_snode(axis_seq, shape_seq, same_level).place(entries_grad, offset=offset)
            if needs_dual:
                impl._create_snode(axis_seq, shape_seq, same_level).place(entries_dual, offset=offset)
        return entries

    @classmethod
    @python_scope
    def ndarray(self, **kwargs):
        kwargs["ndim"] = self.ndim
        return Matrix.ndarray(self.n, self.m, dtype=self.dtype, **kwargs)

    def get_shape(self):
        return (self.n,) if self.ndim == 1 else (self.n, self.m)

    def to_string(self):
        return f"MatrixType[{self.n},{self.m},{self.dtype.to_string() if self.dtype else ''}]"

    def check_matched(self, other):
        return (
            self.ndim == len(other.shape())
            and (self.dtype is None or self.dtype == other.element_type())
            and all(s is None or s == other.shape()[i] for i, s in enumerate(self.get_shape()))
        )

    def ndarray(cls, n, m, dtype, shape):
        return MatrixNdarray(n, m, dtype, (shape,) if isinstance(shape, numbers.Number) else shape)

    @staticmethod
    def rows(rows):
        from taichi.lang import matrix_ops
        return matrix_ops.rows(rows)

    @staticmethod
    def cols(cols):
        from taichi.lang import matrix_ops
        return matrix_ops.cols(cols)

    def __hash__(self):
        return id(self)

    def dot(self, other):
        from taichi.lang import matrix_ops
        return matrix_ops.dot(self, other)

    def cross(self, other):
        from taichi.lang import matrix_ops
        return matrix_ops.cross(self, other)

    def outer_product(self, other):
        from taichi.lang import matrix_ops
        return matrix_ops.outer_product(self, other)


class Vector(Matrix):
    def __init__(self, arr, dt=None, **kwargs):
        super().__init__(arr, dt=dt, **kwargs)

    def get_shape(self):
        return (self.n,)

    @classmethod
    def field(cls, n, dtype, *args, **kwargs):
        kwargs["ndim"] = 1
        return super().field(n, 1, dtype, *args, **kwargs)

    @classmethod
    @python_scope
    def ndarray(cls, n, dtype, shape):
        return VectorNdarray(n, dtype, (shape,) if isinstance(shape, numbers.Number) else shape)


class MatrixField(Field):
    def __init__(self, _vars, n, m, ndim=2):
        assert len(_vars) == n * m and ndim in (0, 1, 2)
        super().__init__(_vars)
        self.n, self.m, self.ndim = n, m, ndim
        self.ptr = ti_python_core.expr_matrix_field([v.ptr for v in self.vars], [n, m][:ndim])

    def get_scalar_field(self, *indices):
        i, j = indices[0], 0 if len(indices) == 1 else indices[1]
        return ScalarField(self.vars[i * self.m + j])

    def _get_dynamic_index_stride(self):
        return self.ptr.get_dynamic_index_stride() if self.ptr.get_dynamic_indexable() else None

    def _calc_dynamic_index_stride(self):
        paths = [ScalarField(var).snode._path_from_root() for var in self.vars]
        if len(paths) == 1:
            self.ptr.set_dynamic_index_stride(0)
            return
        length = len(paths[0])
        if any(len(p) != length or ti_python_core.is_quant(p[length - 1]._dtype) for p in paths):
            return
        for i in range(length):
            if any(p[i] != paths[0][i] for p in paths):
                depth_below_lca = i
                break
        for i in range(depth_below_lca, length - 1):
            if any(
                p[i].ptr.type != ti_python_core.SNodeType.dense
                or p[i]._cell_size_bytes != paths[0][i]._cell_size_bytes
                or p[i + 1]._offset_bytes_in_parent_cell != paths[0][i + 1]._offset_bytes_in_parent_cell
                for p in paths
            ):
                return
        stride = (
            paths[1][depth_below_lca]._offset_bytes_in_parent_cell
            - paths[0][depth_below_lca]._offset_bytes_in_parent_cell
        )
        for i in range(2, len(paths)):
            if stride != (
                paths[i][depth_below_lca]._offset_bytes_in_parent_cell
                - paths[i - 1][depth_below_lca]._offset_bytes_in_parent_cell
            ):
                return
        self.ptr.set_dynamic_index_stride(stride)

    def fill(self, val):
        if isinstance(val, numbers.Number) or (isinstance(val, expr.Expr) and not val.is_tensor()):
            val = (
                tuple(tuple(val for _ in range(self.m)) for _ in range(self.n))
                if self.ndim == 2
                else tuple(val for _ in range(self.n))
            )
        elif isinstance(val, expr.Expr) and val.is_tensor():
            assert val.n == self.n and (self.ndim == 1 or val.m == self.m)
        else:
            val = val.to_list() if isinstance(val, Matrix) else val
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
            assert len(val) == self.n and (self.ndim == 1 or len(val[0]) == self.m)

        if in_python_scope():
            from taichi._kernels import field_fill_python_scope
            field_fill_python_scope(self, val)
        else:
            from taichi._funcs import field_fill_taichi_scope
            field_fill_taichi_scope(self, val)

    @python_scope
    def to_numpy(self, keep_dims=False, dtype=None):
        dtype = to_numpy_type(self.dtype) if dtype is None else dtype
        as_vector = self.m == 1 and not keep_dims
        arr = np.zeros(self.shape + ((self.n,) if as_vector else (self.n, self.m)), dtype=dtype)
        from taichi._kernels import matrix_to_ext_arr
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    def to_torch(self, device=None, keep_dims=False):
        import torch
        as_vector = self.m == 1 and not keep_dims
        arr = torch.empty(
            self.shape + ((self.n,) if as_vector else (self.n, self.m)),
            dtype=to_pytorch_type(self.dtype),
            device=device,
        )
        from taichi._kernels import matrix_to_ext_arr
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    def to_paddle(self, place=None, keep_dims=False):
        import paddle  # pyright: ignore[reportMissingImports]        except ImportError:
        as_vector = self.m == 1 and not keep_dims and self.ndim == 1
        arr = paddle.to_tensor(
            paddle.empty(
                self.shape + ((self.n,) if as_vector else (self.n, self.m)),
                to_paddle_type(self.dtype),
            ),
            place=place,
        )
        from taichi._kernels import matrix_to_ext_arr
        matrix_to_ext_arr(self, arr, as_vector)
        runtime_ops.sync()
        return arr

    @python_scope
    def _from_external_arr(self, arr):
        as_vector = len(arr.shape) == len(self.shape) + 1
        assert as_vector or len(arr.shape) == len(self.shape) + 2
        if as_vector:
            assert self.m == 1
        from taichi._kernels import ext_arr_to_matrix
        ext_arr_to_matrix(arr, self, as_vector)
        runtime_ops.sync()

    @python_scope
    def from_numpy(self, arr):
        self._from_external_arr(np.ascontiguousarray(arr) if not arr.flags.c_contiguous else arr)

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self[key]._set_entries(value)

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        key = self._pad_key(key)
        ha = self._host_access(key)
        return (
            Vector([ha[i] for i in range(self.n)])
            if self.ndim == 1
            else Matrix([[ha[i * self.m + j] for j in range(self.m)] for i in range(self.n)])
        )

    def __repr__(self):
        return f"<{self.n}x{self.m} ti.Matrix.field>"


class MatrixType(CompoundType):
    def __init__(self, n, m, ndim, dtype):
        self.n, self.m, self.ndim = n, m, ndim
        if dtype:
            self.dtype = cook_dtype(dtype)
            self.tensor_type = _type_factory.get_tensor_type((n, m) if ndim == 2 else (n,), self.dtype)
        else:
            self.dtype = self.tensor_type = None

    def __call__(self, *args):
        if not args:
            raise TaichiSyntaxError("Need initial value")
        if len(args) == 1:
            if isinstance(args[0], expr.Expr) and args[0].ptr.is_tensor():
                shape = args[0].ptr.get_rvalue_type().shape()
                assert self.ndim == len(shape) and self.n == shape[0] and (self.ndim == 1 or self.m == shape[1])
                return expr.Expr(args[0].ptr)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                return self._instantiate([args[0]] * (self.m * self.n))
            args = args[0]

        entries = []
        for x in args:
            entries += (
                x
                if isinstance(x, (list, tuple))
                else list(x.ravel())
                if isinstance(x, np.ndarray)
                else x.to_list()
                if isinstance(x, Matrix)
                else [x]
            )
        return self._instantiate(entries)

    def from_taichi_object(self, func_ret, ret_index=()):
        return self(
            [
                expr.Expr(
                    ti_python_core.make_get_element_expr(
                        func_ret.ptr,
                        ret_index + (i,),
                        _ti_python_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                    )
                )
                for i in range(self.m * self.n)
            ]
        )

    def from_kernel_struct_ret(self, launch_ctx, ret_index=()):
        get_ret = (
            launch_ctx.get_struct_ret_int
            if self.dtype in primitive_types.integer_types and is_signed(cook_dtype(self.dtype))
            else launch_ctx.get_struct_ret_uint
            if self.dtype in primitive_types.integer_types
            else launch_ctx.get_struct_ret_float
            if self.dtype in primitive_types.real_types
            else None
        )
        if not get_ret:
            raise TaichiRuntimeTypeError(f"Invalid return type at {ret_index}")
        return self([get_ret(ret_index + (i,)) for i in range(self.m * self.n)])

    def set_kernel_struct_args(self, mat, launch_ctx, ret_index=()):
        set_arg = (
            launch_ctx.set_struct_arg_int
            if self.dtype in primitive_types.integer_types and is_signed(cook_dtype(self.dtype))
            else launch_ctx.set_struct_arg_uint
            if self.dtype in primitive_types.integer_types
            else launch_ctx.set_struct_arg_float
            if self.dtype in primitive_types.real_types
            else None
        )
        if not set_arg:
            raise TaichiRuntimeTypeError(f"Invalid type at {ret_index}")
        if self.ndim == 1:
            for i in range(self.n):
                set_arg(ret_index + (i,), mat[i])
        else:
            for i in range(self.n):
                for j in range(self.m):
                    set_arg(ret_index + (i * self.m + j,), mat[i, j])

    def set_argpack_struct_args(self, mat, argpack, ret_index=()):
        set_arg = (
            argpack.set_arg_int
            if self.dtype in primitive_types.integer_types and is_signed(cook_dtype(self.dtype))
            else argpack.set_arg_uint
            if self.dtype in primitive_types.integer_types
            else argpack.set_arg_float
            if self.dtype in primitive_types.real_types
            else None
        )
        if not set_arg:
            raise TaichiRuntimeTypeError(f"Invalid type at {ret_index}")
        if self.ndim == 1:
            for i in range(self.n):
                set_arg(ret_index + (i,), mat[i])
        else:
            for i in range(self.n):
                for j in range(self.m):
                    set_arg(ret_index + (i * self.m + j,), mat[i, j])

    def _instantiate_in_python_scope(self, entries):
        conv = int if self.dtype in primitive_types.integer_types else float
        return Matrix([[conv(entries[k * self.m + i]) for i in range(self.m)] for k in range(self.n)], dt=self.dtype)

    def _instantiate(self, entries):
        return self._instantiate_in_python_scope(entries) if in_python_scope() else make_matrix_with_shape(entries, [self.n, self.m], self.dtype)

    def field(self, **kwargs):
        kwargs["ndim"] = self.ndim
        return Matrix.field(self.n, self.m, dtype=self.dtype, **kwargs)

    def ndarray(self, **kwargs):
        kwargs["ndim"] = self.ndim
        return Matrix.ndarray(self.n, self.m, dtype=self.dtype, **kwargs)

    def get_shape(self):
        return (self.n,) if self.ndim == 1 else (self.n, self.m)

    def to_string(self):
        return f"MatrixType[{self.n},{self.m},{self.dtype.to_string() if self.dtype else ''}]"

    def check_matched(self, other):
        return (
            self.ndim == len(other.shape())
            and (self.dtype is None or self.dtype == other.element_type())
            and all(s is None or s == other.shape()[i] for i, s in enumerate(self.get_shape()))
        )


class VectorType(MatrixType):
    def __init__(self, n, dtype):
        super().__init__(n, 1, 1, dtype)

    def __call__(self, *args):
        if not args:
            raise TaichiSyntaxError("Need initial value")
        if len(args) == 1:
            if isinstance(args[0], expr.Expr) and args[0].ptr.is_tensor():
                assert len(args[0].ptr.get_rvalue_type().shape()) == 1 and self.n == args[0].ptr.get_rvalue_type().shape()[0]
                return expr.Expr(args[0].ptr)
            if isinstance(args[0], (numbers.Number, expr.Expr)):
                return self._instantiate([args[0]] * self.n)
            args = args[0]

        entries = []
        for x in args:
            entries += (
                x
                if isinstance(x, (list, tuple))
                else list(x.ravel())
                if isinstance(x, np.ndarray)
                else x.to_list()
                if isinstance(x, Matrix)
                else [x]
            )
        return self._instantiate(entries)

    def _instantiate_in_python_scope(self, entries):
        conv = int if self.dtype in primitive_types.integer_types else float
        return Vector([conv(entries[i]) for i in range(self.n)], dt=self.dtype)

    def _instantiate(self, entries):
        return self._instantiate_in_python_scope(entries) if in_python_scope() else make_matrix_with_shape(entries, [self.n], self.dtype)

    def field(self, **kwargs):
        return Vector.field(self.n, dtype=self.dtype, **kwargs)

    def ndarray(self, **kwargs):
        return Vector.ndarray(self.n, dtype=self.dtype, **kwargs)

    def to_string(self):
        return f"VectorType[{self.n},{self.dtype.to_string() if self.dtype else ''}]"


class MatrixNdarray(Ndarray):
    def __init__(self, n, m, dtype, shape):
        self.n, self.m = n, m
        super().__init__()
        self.dtype, self.layout, self.shape = cook_dtype(dtype), Layout.AOS, tuple(shape)
        self.element_type = _type_factory.get_tensor_type((self.n, self.m), self.dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type),
            shape,
            Layout.AOS,
            zero_fill=True,
            dbg_info=ti_python_core.DebugInfo(get_traceback()),
        )

    @property
    def element_shape(self):
        return tuple(self.arr.element_shape())

    @python_scope
    def __setitem__(self, key, value):
        value = list(value) if not isinstance(value, (list, tuple)) else value
        value = [[i] for i in value] if not isinstance(value[0], (list, tuple)) else value
        for i in range(self.n):
            for j in range(self.m):
                self[key][i, j] = value[i][j]

    @python_scope
    def __getitem__(self, key):
        key = () if key is None else (key,) if isinstance(key, numbers.Number) else tuple(key)
        return Matrix([[NdarrayHostAccess(self, key, (i, j)) for j in range(self.m)] for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        return self._ndarray_matrix_to_numpy(as_vector=0)

    @python_scope
    def from_numpy(self, arr):
        self._ndarray_matrix_from_numpy(arr, as_vector=0)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret = MatrixNdarray(self.n, self.m, self.dtype, self.shape)
        ret.copy_from(self)
        return ret

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import fill_ndarray_matrix
        shape = self.element_type.shape()
        mat_type = MatrixType(
            shape[0], 1 if len(shape) == 1 else shape[1], len(shape), self.element_type.element_type()
        )
        fill_ndarray_matrix(self, val if isinstance(val, Matrix) else mat_type(val))

    @python_scope
    def __repr__(self):
        return f"<{self.n}x{self.m} {Layout.AOS} ti.Matrix.ndarray>"


class VectorNdarray(Ndarray):
    def __init__(self, n, dtype, shape):
        self.n = n
        super().__init__()
        self.dtype, self.layout, self.shape = cook_dtype(dtype), Layout.AOS, tuple(shape)
        self.element_type = _type_factory.get_tensor_type((n,), self.dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            cook_dtype(self.element_type),
            shape,
            Layout.AOS,
            zero_fill=True,
            dbg_info=ti_python_core.DebugInfo(get_traceback()),
        )

    @property
    def element_shape(self):
        return tuple(self.arr.element_shape())

    @python_scope
    def __setitem__(self, key, value):
        value = list(value) if not isinstance(value, (list, tuple)) else value
        for i in range(self.n):
            self[key][i] = value[i]

    @python_scope
    def __getitem__(self, key):
        key = () if key is None else (key,) if isinstance(key, numbers.Number) else tuple(key)
        return Vector([NdarrayHostAccess(self, key, (i,)) for i in range(self.n)])

    @python_scope
    def to_numpy(self):
        return self._ndarray_matrix_to_numpy(as_vector=1)

    @python_scope
    def from_numpy(self, arr):
        self._ndarray_matrix_from_numpy(arr, as_vector=1)

    @python_scope
    def __deepcopy__(self, memo=None):
        ret = VectorNdarray(self.n, self.dtype, self.shape)
        ret.copy_from(self)
        return ret

    @python_scope
    def _fill_by_kernel(self, val):
        from taichi._kernels import fill_ndarray_matrix
        vec_type = VectorType(self.element_type.shape()[0], self.element_type.element_type())
        fill_ndarray_matrix(self, val if isinstance(val, Vector) else vec_type(val))

    @python_scope
    def __repr__(self):
        return f"<{self.n} {Layout.AOS} ti.Vector.ndarray>"


__all__ = ["Matrix", "Vector", "MatrixField", "MatrixNdarray", "VectorNdarray"]
