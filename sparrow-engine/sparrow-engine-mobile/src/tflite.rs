//! LiteRT/TensorFlow Lite backend wrapper for the mobile engine flavor.

use crate::sys;
use anyhow::{anyhow, bail, Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

/// LiteRT tensor element type accepted by [`LiteRtBackend::invoke_named`].
pub type ElementType = sys::LiteRtElementType;

/// One LiteRT-loaded model using the CPU backend.
///
/// The wrapper owns the LiteRT environment, model options, compiled model, and
/// model handle. It is intentionally minimal for P2.1: no sparrow-engine manifest
/// loading or preprocessing is wired here yet.
pub struct LiteRtBackend {
    env: sys::LiteRtEnvironment,
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
    /// Load and compile a TFLite/LiteRT model file for CPU inference.
    ///
    /// `num_threads == 0` leaves LiteRT's default CPU thread count unchanged.
    pub fn load(path: &Path, num_threads: usize) -> Result<Self> {
        let path_cstr = CString::new(path.to_str().context("model path must be utf-8")?)?;

        unsafe {
            let mut env: sys::LiteRtEnvironment = ptr::null_mut();
            check(
                sys::LiteRtCreateEnvironment(0, ptr::null(), &mut env),
                "LiteRtCreateEnvironment",
            )?;

            let mut model: sys::LiteRtModel = ptr::null_mut();
            check(
                sys::LiteRtCreateModelFromFile(path_cstr.as_ptr(), &mut model),
                "LiteRtCreateModelFromFile",
            )?;

            let mut opts: sys::LiteRtOptions = ptr::null_mut();
            check(sys::LiteRtCreateOptions(&mut opts), "LiteRtCreateOptions")?;
            check(
                sys::LiteRtSetOptionsHardwareAccelerators(
                    opts,
                    sys::LiteRtHwAccelerators::kLiteRtHwAcceleratorCpu
                        as sys::LiteRtHwAcceleratorSet,
                ),
                "LiteRtSetOptionsHardwareAccelerators",
            )?;

            if num_threads > 0 {
                tracing::debug!(num_threads, "setting LiteRT CPU inference thread count");
                let mut cpu_opts: *mut sys::LrtCpuOptions = ptr::null_mut();
                check(
                    sys::LrtCreateCpuOptions(&mut cpu_opts),
                    "LrtCreateCpuOptions",
                )?;
                check(
                    sys::LrtSetCpuOptionsNumThread(cpu_opts, num_threads as c_int),
                    "LrtSetCpuOptionsNumThread",
                )?;
                let mut id: *const c_char = ptr::null();
                let mut payload: *mut c_void = ptr::null_mut();
                let mut deleter: Option<unsafe extern "C" fn(*mut c_void)> = None;
                check(
                    sys::LrtGetOpaqueCpuOptionsData(cpu_opts, &mut id, &mut payload, &mut deleter),
                    "LrtGetOpaqueCpuOptionsData",
                )?;
                let mut opaque: sys::LiteRtOpaqueOptions = ptr::null_mut();
                check(
                    sys::LiteRtCreateOpaqueOptions(id, payload, deleter, &mut opaque),
                    "LiteRtCreateOpaqueOptions",
                )?;
                check(
                    sys::LiteRtAddOpaqueOptions(opts, opaque),
                    "LiteRtAddOpaqueOptions",
                )?;
                sys::LrtDestroyCpuOptions(cpu_opts);
            }

            let mut compiled: sys::LiteRtCompiledModel = ptr::null_mut();
            check(
                sys::LiteRtCreateCompiledModel(env, model, opts, &mut compiled),
                "LiteRtCreateCompiledModel",
            )?;

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

            Ok(Self {
                env,
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

    /// Model signature input names, in LiteRT slot order.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Number of model outputs.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
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

        unsafe {
            let mut in_bufs: Vec<sys::LiteRtTensorBuffer> = Vec::with_capacity(self.num_inputs);
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
                let mut buf: sys::LiteRtTensorBuffer = ptr::null_mut();
                check(
                    sys::LiteRtCreateManagedTensorBufferFromRequirements(
                        self.env,
                        &tensor_type,
                        req,
                        &mut buf,
                    ),
                    "LiteRtCreateManagedTensorBufferFromRequirements(input)",
                )?;
                let mut host_ptr: *mut c_void = ptr::null_mut();
                check(
                    sys::LiteRtLockTensorBuffer(
                        buf,
                        &mut host_ptr,
                        sys::LiteRtTensorBufferLockMode::kLiteRtTensorBufferLockModeWrite,
                    ),
                    "LiteRtLockTensorBuffer(input write)",
                )?;
                ptr::copy_nonoverlapping(bytes.as_ptr(), host_ptr as *mut u8, bytes.len());
                check(
                    sys::LiteRtUnlockTensorBuffer(buf),
                    "LiteRtUnlockTensorBuffer(input)",
                )?;
                in_bufs.push(buf);
            }

            let mut out_bufs: Vec<sys::LiteRtTensorBuffer> = Vec::with_capacity(self.num_outputs);
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
                        self.env,
                        &tensor_type,
                        req,
                        &mut buf,
                    ),
                    "LiteRtCreateManagedTensorBufferFromRequirements(output)",
                )?;
                out_bufs.push(buf);
            }

            check(
                sys::LiteRtRunCompiledModel(
                    self.compiled,
                    0,
                    in_bufs.len(),
                    in_bufs.as_mut_ptr(),
                    out_bufs.len(),
                    out_bufs.as_mut_ptr(),
                ),
                "LiteRtRunCompiledModel",
            )?;

            let mut outs: Vec<Vec<f32>> = Vec::with_capacity(self.num_outputs);
            for (i, buf) in out_bufs.iter().enumerate() {
                let n_elems = layout_num_elements(&self.output_layouts[i])?;
                let mut host_ptr: *mut c_void = ptr::null_mut();
                check(
                    sys::LiteRtLockTensorBuffer(
                        *buf,
                        &mut host_ptr,
                        sys::LiteRtTensorBufferLockMode::kLiteRtTensorBufferLockModeRead,
                    ),
                    "LiteRtLockTensorBuffer(output)",
                )?;
                let slice = std::slice::from_raw_parts(host_ptr as *const f32, n_elems);
                outs.push(slice.to_vec());
                check(
                    sys::LiteRtUnlockTensorBuffer(*buf),
                    "LiteRtUnlockTensorBuffer(output)",
                )?;
            }

            for buf in in_bufs {
                sys::LiteRtDestroyTensorBuffer(buf);
            }
            for buf in out_bufs {
                sys::LiteRtDestroyTensorBuffer(buf);
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
            if !self.env.is_null() {
                sys::LiteRtDestroyEnvironment(self.env);
            }
        }
    }
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
