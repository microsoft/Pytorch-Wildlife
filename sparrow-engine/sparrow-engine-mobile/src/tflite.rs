//! LiteRT/TensorFlow Lite backend wrapper for the mobile engine flavor.

use crate::sys;
use crate::timing;
use anyhow::{anyhow, bail, Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::time::Instant;

/// LiteRT tensor element type accepted by [`LiteRtBackend::invoke_single`] /
/// [`LiteRtBackend::invoke_named`].
pub type ElementType = sys::LiteRtElementType;

/// Process-local LiteRT runtime.
///
/// Owns one `LiteRtEnvironment` and shares it with every compiled model loaded
/// through [`LiteRtRuntime::load`]. The backend holds an `Rc` clone of the
/// runtime inner state so the environment outlives all compiled models.
#[derive(Clone)]
pub struct LiteRtRuntime {
    inner: Rc<LiteRtRuntimeInner>,
}

struct LiteRtRuntimeInner {
    env: sys::LiteRtEnvironment,
}

impl LiteRtRuntime {
    /// Create one LiteRT environment for all models in this process/cascade.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut env: sys::LiteRtEnvironment = ptr::null_mut();
            check(
                sys::LiteRtCreateEnvironment(0, ptr::null(), &mut env),
                "LiteRtCreateEnvironment",
            )?;
            Ok(Self {
                inner: Rc::new(LiteRtRuntimeInner { env }),
            })
        }
    }

    /// Load and compile a TFLite/LiteRT model file for CPU inference.
    ///
    /// `num_threads == 0` leaves LiteRT's default CPU thread count unchanged.
    pub fn load(&self, path: &Path, num_threads: usize) -> Result<LiteRtBackend> {
        LiteRtBackend::load_with_runtime(self.inner.clone(), path, num_threads)
    }
}

impl Drop for LiteRtRuntimeInner {
    fn drop(&mut self) {
        unsafe {
            if !self.env.is_null() {
                sys::LiteRtDestroyEnvironment(self.env);
            }
        }
    }
}

/// One LiteRT-loaded model using the CPU backend.
///
/// The wrapper owns model options, compiled model, and model handle. It borrows
/// the shared LiteRT environment by holding an `Rc` to the runtime inner state;
/// it never destroys the environment itself.
pub struct LiteRtBackend {
    runtime: Rc<LiteRtRuntimeInner>,
    model: sys::LiteRtModel,
    opts: sys::LiteRtOptions,
    compiled: sys::LiteRtCompiledModel,
    num_inputs: usize,
    num_outputs: usize,
    input_layouts: Vec<sys::LiteRtLayout>,
    output_layouts: Vec<sys::LiteRtLayout>,
    input_names: Vec<String>,
}

impl LiteRtBackend {
    fn load_with_runtime(
        runtime: Rc<LiteRtRuntimeInner>,
        path: &Path,
        num_threads: usize,
    ) -> Result<Self> {
        let path_cstr = CString::new(path.to_str().context("model path must be utf-8")?)?;

        unsafe {
            let mut model: sys::LiteRtModel = ptr::null_mut();
            check(
                sys::LiteRtCreateModelFromFile(path_cstr.as_ptr(), &mut model),
                "LiteRtCreateModelFromFile",
            )?;
            // Guard the raw handles so any `?` below frees them on early return; the
            // guard is mem::forget-disarmed once they are moved into `Self` (whose Drop
            // then frees each exactly once — no double-free).
            let mut guard = LoadGuard {
                model,
                opts: ptr::null_mut(),
                compiled: ptr::null_mut(),
            };

            let mut opts: sys::LiteRtOptions = ptr::null_mut();
            check(sys::LiteRtCreateOptions(&mut opts), "LiteRtCreateOptions")?;
            guard.opts = opts;
            check(
                sys::LiteRtSetOptionsHardwareAccelerators(
                    opts,
                    sys::LiteRtHwAccelerators::kLiteRtHwAcceleratorCpu
                        as sys::LiteRtHwAcceleratorSet,
                ),
                "LiteRtSetOptionsHardwareAccelerators",
            )?;

            if num_threads > 0 {
                attach_cpu_thread_options(opts, num_threads)?;
            }

            let mut compiled: sys::LiteRtCompiledModel = ptr::null_mut();
            check(
                sys::LiteRtCreateCompiledModel(runtime.env, model, opts, &mut compiled),
                "LiteRtCreateCompiledModel",
            )?;
            guard.compiled = compiled;

            let mut sig: sys::LiteRtSignature = ptr::null_mut();
            check(
                sys::LiteRtGetModelSignature(model, 0, &mut sig),
                "LiteRtGetModelSignature(0)",
            )?;
            let mut num_inputs: sys::LiteRtParamIndex = 0;
            check(
                sys::LiteRtGetNumSignatureInputs(sig, &mut num_inputs),
                "LiteRtGetNumSignatureInputs",
            )?;
            let mut num_outputs: sys::LiteRtParamIndex = 0;
            check(
                sys::LiteRtGetNumSignatureOutputs(sig, &mut num_outputs),
                "LiteRtGetNumSignatureOutputs",
            )?;

            let mut input_layouts = Vec::with_capacity(num_inputs as usize);
            let mut input_names = Vec::with_capacity(num_inputs as usize);
            for i in 0..num_inputs {
                let mut layout: sys::LiteRtLayout = std::mem::zeroed();
                check(
                    sys::LiteRtGetCompiledModelInputTensorLayout(compiled, 0, i, &mut layout),
                    "LiteRtGetCompiledModelInputTensorLayout",
                )?;
                input_layouts.push(layout);

                let mut name_ptr: *const c_char = ptr::null();
                check(
                    sys::LiteRtGetSignatureInputName(sig, i, &mut name_ptr),
                    "LiteRtGetSignatureInputName",
                )?;
                input_names.push(CStr::from_ptr(name_ptr).to_string_lossy().into_owned());
            }

            let mut output_layouts: Vec<sys::LiteRtLayout> =
                vec![std::mem::zeroed(); num_outputs as usize];
            check(
                sys::LiteRtGetCompiledModelOutputTensorLayouts(
                    compiled,
                    0,
                    num_outputs as usize,
                    output_layouts.as_mut_ptr(),
                    false,
                ),
                "LiteRtGetCompiledModelOutputTensorLayouts",
            )?;

            // Success: `Self` now owns the handles and frees them on Drop. Disarm the
            // guard so they are not freed twice.
            std::mem::forget(guard);
            Ok(Self {
                runtime,
                model,
                opts,
                compiled,
                num_inputs: num_inputs as usize,
                num_outputs: num_outputs as usize,
                input_layouts,
                output_layouts,
                input_names,
            })
        }
    }

    /// Run inference, routing each input by a substring of its signature name.
    pub fn invoke_named(
        &mut self,
        named: &[(&str, Vec<u8>, ElementType)],
    ) -> Result<Vec<Vec<f32>>> {
        if named.len() != self.num_inputs {
            bail!(
                "invoke_named arity mismatch: model expects {} input(s), got {}",
                self.num_inputs,
                named.len()
            );
        }
        let mut routed: Vec<(Vec<u8>, sys::LiteRtElementType)> =
            vec![(Vec::new(), sys::LiteRtElementType::kLiteRtElementTypeNone); self.num_inputs];
        for (needle, bytes, etype) in named {
            let idx = self.find_input(needle)?;
            routed[idx] = (bytes.clone(), *etype);
        }
        self.invoke(&routed)
    }

    /// Run inference on a single-input model, feeding `bytes` to input 0.
    ///
    /// Generic-engine path: most sparrow-engine models (image detectors,
    /// classifiers, mel-input audio models) take exactly one input tensor, so the
    /// name-substring routing of [`invoke_named`](Self::invoke_named) is
    /// unnecessary. Errors if the model does not have exactly one input.
    pub fn invoke_single(
        &mut self,
        bytes: Vec<u8>,
        etype: ElementType,
    ) -> Result<Vec<Vec<f32>>> {
        if self.num_inputs != 1 {
            bail!(
                "invoke_single requires a single-input model, but this model has {} inputs ({:?})",
                self.num_inputs,
                self.input_names
            );
        }
        self.invoke(&[(bytes, etype)])
    }

    fn find_input(&self, needle: &str) -> Result<usize> {
        for (i, name) in self.input_names.iter().enumerate() {
            if name.contains(needle) {
                return Ok(i);
            }
        }
        bail!(
            "input '{needle}' not found in model signature; have: {:?}",
            self.input_names
        );
    }

    fn invoke(&mut self, inputs: &[(Vec<u8>, sys::LiteRtElementType)]) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != self.num_inputs {
            bail!(
                "invoke arity mismatch: model expects {} input(s), got {}",
                self.num_inputs,
                inputs.len()
            );
        }

        let timed = timing::enabled();
        unsafe {
            let t_setup = Instant::now();
            let mut in_bufs: Vec<OwnedTensorBuffer> = Vec::with_capacity(self.num_inputs);
            for (i, (bytes, etype)) in inputs.iter().enumerate() {
                let mut req: sys::LiteRtTensorBufferRequirements = ptr::null_mut();
                check(
                    sys::LiteRtGetCompiledModelInputBufferRequirements(
                        self.compiled,
                        0,
                        i as sys::LiteRtParamIndex,
                        &mut req,
                    ),
                    "LiteRtGetCompiledModelInputBufferRequirements",
                )?;
                let tensor_type = sys::LiteRtRankedTensorType {
                    element_type: *etype,
                    layout: self.input_layouts[i],
                };
                let expected_bytes =
                    layout_num_elements(&self.input_layouts[i])? * element_type_byte_len(*etype)?;
                if bytes.len() != expected_bytes {
                    bail!(
                        "input {i} byte length mismatch: got {}, expected {} for {:?}",
                        bytes.len(),
                        expected_bytes,
                        etype
                    );
                }
                let mut buf: sys::LiteRtTensorBuffer = ptr::null_mut();
                check(
                    sys::LiteRtCreateManagedTensorBufferFromRequirements(
                        self.runtime.env,
                        &tensor_type,
                        req,
                        &mut buf,
                    ),
                    "LiteRtCreateManagedTensorBufferFromRequirements(input)",
                )?;
                let buf = OwnedTensorBuffer::new(buf);
                let mut host_ptr: *mut c_void = ptr::null_mut();
                check(
                    sys::LiteRtLockTensorBuffer(
                        buf.as_raw(),
                        &mut host_ptr,
                        sys::LiteRtTensorBufferLockMode::kLiteRtTensorBufferLockModeWrite,
                    ),
                    "LiteRtLockTensorBuffer(input write)",
                )?;
                let lock = LockedTensorBuffer::new(buf.as_raw());
                ptr::copy_nonoverlapping(bytes.as_ptr(), host_ptr as *mut u8, bytes.len());
                lock.unlock("LiteRtUnlockTensorBuffer(input)")?;
                in_bufs.push(buf);
            }

            let mut out_bufs: Vec<OwnedTensorBuffer> = Vec::with_capacity(self.num_outputs);
            for i in 0..self.num_outputs {
                let mut req: sys::LiteRtTensorBufferRequirements = ptr::null_mut();
                check(
                    sys::LiteRtGetCompiledModelOutputBufferRequirements(
                        self.compiled,
                        0,
                        i as sys::LiteRtParamIndex,
                        &mut req,
                    ),
                    "LiteRtGetCompiledModelOutputBufferRequirements",
                )?;
                let tensor_type = sys::LiteRtRankedTensorType {
                    element_type: sys::LiteRtElementType::kLiteRtElementTypeFloat32,
                    layout: self.output_layouts[i],
                };
                let mut buf: sys::LiteRtTensorBuffer = ptr::null_mut();
                check(
                    sys::LiteRtCreateManagedTensorBufferFromRequirements(
                        self.runtime.env,
                        &tensor_type,
                        req,
                        &mut buf,
                    ),
                    "LiteRtCreateManagedTensorBufferFromRequirements(output)",
                )?;
                out_bufs.push(OwnedTensorBuffer::new(buf));
            }

            let mut in_raws: Vec<sys::LiteRtTensorBuffer> =
                in_bufs.iter().map(OwnedTensorBuffer::as_raw).collect();
            let mut out_raws: Vec<sys::LiteRtTensorBuffer> =
                out_bufs.iter().map(OwnedTensorBuffer::as_raw).collect();
            if timed {
                timing::add_setup(t_setup.elapsed().as_nanos());
            }
            let t_run = Instant::now();
            check(
                sys::LiteRtRunCompiledModel(
                    self.compiled,
                    0,
                    in_raws.len(),
                    in_raws.as_mut_ptr(),
                    out_raws.len(),
                    out_raws.as_mut_ptr(),
                ),
                "LiteRtRunCompiledModel",
            )?;
            if timed {
                timing::add_run(t_run.elapsed().as_nanos());
            }
            let t_read = Instant::now();

            let mut outs: Vec<Vec<f32>> = Vec::with_capacity(self.num_outputs);
            for (i, buf) in out_bufs.iter().enumerate() {
                let n_elems = layout_num_elements(&self.output_layouts[i])?;
                let mut host_ptr: *mut c_void = ptr::null_mut();
                check(
                    sys::LiteRtLockTensorBuffer(
                        buf.as_raw(),
                        &mut host_ptr,
                        sys::LiteRtTensorBufferLockMode::kLiteRtTensorBufferLockModeRead,
                    ),
                    "LiteRtLockTensorBuffer(output)",
                )?;
                let lock = LockedTensorBuffer::new(buf.as_raw());
                let slice = std::slice::from_raw_parts(host_ptr as *const f32, n_elems);
                outs.push(slice.to_vec());
                lock.unlock("LiteRtUnlockTensorBuffer(output)")?;
            }
            if timed {
                timing::add_read(t_read.elapsed().as_nanos());
            }

            Ok(outs)
        }
    }
}

impl Drop for LiteRtBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.compiled.is_null() {
                sys::LiteRtDestroyCompiledModel(self.compiled);
            }
            if !self.opts.is_null() {
                sys::LiteRtDestroyOptions(self.opts);
            }
            if !self.model.is_null() {
                sys::LiteRtDestroyModel(self.model);
            }
        }
    }
}

/// RAII guard for the raw LiteRT handles during `LiteRtBackend::load_with_runtime`.
/// `LiteRtBackend::drop` only runs once a `Self` is constructed; any `?` early-return
/// before that point would otherwise leak `model`/`opts`/`compiled`. This guard frees
/// whichever handles have been created so far, and is `mem::forget`-disarmed on the
/// success path so the constructed backend remains the sole owner (no double-free).
struct LoadGuard {
    model: sys::LiteRtModel,
    opts: sys::LiteRtOptions,
    compiled: sys::LiteRtCompiledModel,
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.compiled.is_null() {
                sys::LiteRtDestroyCompiledModel(self.compiled);
            }
            if !self.opts.is_null() {
                sys::LiteRtDestroyOptions(self.opts);
            }
            if !self.model.is_null() {
                sys::LiteRtDestroyModel(self.model);
            }
        }
    }
}

struct OwnedTensorBuffer(sys::LiteRtTensorBuffer);

impl OwnedTensorBuffer {
    fn new(buffer: sys::LiteRtTensorBuffer) -> Self {
        Self(buffer)
    }

    fn as_raw(&self) -> sys::LiteRtTensorBuffer {
        self.0
    }
}

impl Drop for OwnedTensorBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                sys::LiteRtDestroyTensorBuffer(self.0);
            }
        }
    }
}

struct LockedTensorBuffer {
    buffer: sys::LiteRtTensorBuffer,
    locked: bool,
}

impl LockedTensorBuffer {
    fn new(buffer: sys::LiteRtTensorBuffer) -> Self {
        Self {
            buffer,
            locked: true,
        }
    }

    fn unlock(mut self, context: &str) -> Result<()> {
        let result = unsafe { check(sys::LiteRtUnlockTensorBuffer(self.buffer), context) };
        if result.is_ok() {
            self.locked = false;
        }
        result
    }
}

impl Drop for LockedTensorBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.locked && !self.buffer.is_null() {
                let _ = sys::LiteRtUnlockTensorBuffer(self.buffer);
            }
        }
    }
}

struct CpuOptionsGuard {
    ptr: *mut sys::LrtCpuOptions,
    destroy: unsafe extern "C" fn(*mut sys::LrtCpuOptions),
}

impl CpuOptionsGuard {
    fn new(
        ptr: *mut sys::LrtCpuOptions,
        destroy: unsafe extern "C" fn(*mut sys::LrtCpuOptions),
    ) -> Self {
        Self { ptr, destroy }
    }

    fn as_raw(&self) -> *mut sys::LrtCpuOptions {
        self.ptr
    }
}

impl Drop for CpuOptionsGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                (self.destroy)(self.ptr);
            }
        }
    }
}

fn attach_cpu_thread_options(opts: sys::LiteRtOptions, num_threads: usize) -> Result<()> {
    tracing::debug!(num_threads, "setting LiteRT CPU inference thread count");
    unsafe {
        let symbols = CpuOptionsSymbols::load()?;
        let mut cpu_opts: *mut sys::LrtCpuOptions = ptr::null_mut();
        check((symbols.create)(&mut cpu_opts), "LrtCreateCpuOptions")?;
        let cpu_opts = CpuOptionsGuard::new(cpu_opts, symbols.destroy);
        check(
            (symbols.set_num_threads)(cpu_opts.as_raw(), num_threads as c_int),
            "LrtSetCpuOptionsNumThread",
        )?;
        let mut id: *const c_char = ptr::null();
        let mut payload: *mut c_void = ptr::null_mut();
        let mut deleter: Option<unsafe extern "C" fn(*mut c_void)> = None;
        check(
            (symbols.get_opaque_data)(cpu_opts.as_raw(), &mut id, &mut payload, &mut deleter),
            "LrtGetOpaqueCpuOptionsData",
        )?;
        let mut opaque: sys::LiteRtOpaqueOptions = ptr::null_mut();
        check(
            sys::LiteRtCreateOpaqueOptions(id, payload, deleter, &mut opaque),
            "LiteRtCreateOpaqueOptions",
        )?;
        if let Err(e) = check(
            sys::LiteRtAddOpaqueOptions(opts, opaque),
            "LiteRtAddOpaqueOptions",
        ) {
            sys::LiteRtDestroyOpaqueOptions(opaque);
            return Err(e);
        }
        Ok(())
    }
}

struct CpuOptionsSymbols {
    create: unsafe extern "C" fn(*mut *mut sys::LrtCpuOptions) -> sys::LiteRtStatus,
    destroy: unsafe extern "C" fn(*mut sys::LrtCpuOptions),
    set_num_threads: unsafe extern "C" fn(*mut sys::LrtCpuOptions, c_int) -> sys::LiteRtStatus,
    get_opaque_data: unsafe extern "C" fn(
        *const sys::LrtCpuOptions,
        *mut *const c_char,
        *mut *mut c_void,
        *mut Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> sys::LiteRtStatus,
}

impl CpuOptionsSymbols {
    unsafe fn load() -> Result<Self> {
        Ok(Self {
            create: load_process_symbol(c"LrtCreateCpuOptions")?,
            destroy: load_process_symbol(c"LrtDestroyCpuOptions")?,
            set_num_threads: load_process_symbol(c"LrtSetCpuOptionsNumThread")?,
            get_opaque_data: load_process_symbol(c"LrtGetOpaqueCpuOptionsData")?,
        })
    }
}

#[cfg(unix)]
unsafe fn load_process_symbol<T>(name: &CStr) -> Result<T>
where
    T: Copy,
{
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    let ptr = unsafe { dlsym(ptr::null_mut(), name.as_ptr()) };
    if ptr.is_null() {
        bail!(
            "LiteRT CPU thread symbol '{}' not found; use num_threads=0 with stock host LiteRT",
            name.to_string_lossy()
        );
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

#[cfg(not(unix))]
unsafe fn load_process_symbol<T>(name: &CStr) -> Result<T>
where
    T: Copy,
{
    bail!(
        "dynamic LiteRT CPU thread symbol lookup is not implemented on this platform for '{}'",
        name.to_string_lossy()
    );
}

fn check(status: sys::LiteRtStatus, context: &str) -> Result<()> {
    if status == sys::LiteRtStatus::kLiteRtStatusOk {
        Ok(())
    } else {
        Err(anyhow!("{context} failed: rc={status:?}"))
    }
}

fn layout_num_elements(layout: &sys::LiteRtLayout) -> Result<usize> {
    let rank = layout.rank() as usize;
    if rank > 8 {
        bail!("layout rank {rank} exceeds LITERT_TENSOR_MAX_RANK=8");
    }
    let mut n: usize = 1;
    for i in 0..rank {
        let d = layout.dimensions[i];
        if d < 0 {
            bail!("layout has dynamic dimension at axis {i} (dim={d})");
        }
        n *= d as usize;
    }
    Ok(n)
}

fn element_type_byte_len(element_type: sys::LiteRtElementType) -> Result<usize> {
    let bytes = match element_type {
        sys::LiteRtElementType::kLiteRtElementTypeBool
        | sys::LiteRtElementType::kLiteRtElementTypeInt8
        | sys::LiteRtElementType::kLiteRtElementTypeUInt8 => 1,
        sys::LiteRtElementType::kLiteRtElementTypeInt16
        | sys::LiteRtElementType::kLiteRtElementTypeUInt16
        | sys::LiteRtElementType::kLiteRtElementTypeFloat16
        | sys::LiteRtElementType::kLiteRtElementTypeBFloat16 => 2,
        sys::LiteRtElementType::kLiteRtElementTypeInt32
        | sys::LiteRtElementType::kLiteRtElementTypeUInt32
        | sys::LiteRtElementType::kLiteRtElementTypeFloat32 => 4,
        sys::LiteRtElementType::kLiteRtElementTypeInt64
        | sys::LiteRtElementType::kLiteRtElementTypeUInt64
        | sys::LiteRtElementType::kLiteRtElementTypeFloat64
        | sys::LiteRtElementType::kLiteRtElementTypeComplex64 => 8,
        sys::LiteRtElementType::kLiteRtElementTypeComplex128 => 16,
        other => bail!("unsupported LiteRT input element type {other:?}"),
    };
    Ok(bytes)
}
