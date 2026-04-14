use std::ffi::CString;
use std::os::raw::c_int;
use std::path::Path;

#[repr(C)]
pub struct CQueryResult {
    pub id: u32,
    pub distance: f32,
}

#[repr(C)]
pub struct CIVFSearchParams {
    pub query: *const f32,
    pub d: c_int,
    pub k: c_int,
    pub nprobe: c_int,
    pub refine_factor: c_int,
    pub use_sq8: c_int,
}

#[repr(C)]
pub struct CFlatSearchParams {
    pub query: *const f32,
    pub d: c_int,
    pub k: c_int,
    pub index_type: c_int,
}

extern "C" {
    fn vq_engine_open(path: *const i8) -> *mut std::ffi::c_void;
    fn vq_engine_close(engine: *mut std::ffi::c_void);
    fn vq_ivf_search(
        engine: *mut std::ffi::c_void,
        params: *const CIVFSearchParams,
        n_results: *mut c_int,
    ) -> *mut CQueryResult;
    fn vq_flat_search(
        engine: *mut std::ffi::c_void,
        params: *const CFlatSearchParams,
        n_results: *mut c_int,
    ) -> *mut CQueryResult;
    fn vq_ivf_batch_search(
        engine: *mut std::ffi::c_void,
        queries: *const f32,
        n_queries: c_int,
        d: c_int,
        k: c_int,
        nprobe: c_int,
        refine_factor: c_int,
        use_sq8: c_int,
        n_results_per_query: *mut *mut c_int,
    ) -> *mut CQueryResult;
    fn vq_results_free(results: *mut CQueryResult);
    fn vq_n_results_free(n_results: *mut c_int);
}

pub struct VectorEngine {
    handle: *mut std::ffi::c_void,
}

impl VectorEngine {
    pub fn open(path: &Path) -> Result<Self, String> {
        let c_path = CString::new(path.to_str().ok_or("invalid path")?)
            .map_err(|e| e.to_string())?;
        let handle = unsafe { vq_engine_open(c_path.as_ptr()) };
        if handle.is_null() {
            return Err("failed to open vector engine".into());
        }
        Ok(Self { handle })
    }

    pub fn ivf_search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        refine_factor: usize,
        use_sq8: bool,
    ) -> Vec<(u32, f32)> {
        let d = query.len();
        let params = CIVFSearchParams {
            query: query.as_ptr(),
            d: d as c_int,
            k: k as c_int,
            nprobe: nprobe as c_int,
            refine_factor: refine_factor as c_int,
            use_sq8: use_sq8 as c_int,
        };
        let mut n_results: c_int = 0;
        let results = unsafe { vq_ivf_search(self.handle, &params, &mut n_results) };
        if results.is_null() || n_results == 0 {
            return vec![];
        }
        let vec: Vec<(u32, f32)> = unsafe {
            std::slice::from_raw_parts(results, n_results as usize)
                .iter()
                .map(|r| (r.id, r.distance))
                .collect()
        };
        unsafe { vq_results_free(results) };
        vec
    }

    pub fn flat_search(
        &self,
        query: &[f32],
        k: usize,
        index_type: usize,
    ) -> Vec<(u32, f32)> {
        let d = query.len();
        let params = CFlatSearchParams {
            query: query.as_ptr(),
            d: d as c_int,
            k: k as c_int,
            index_type: index_type as c_int,
        };
        let mut n_results: c_int = 0;
        let results = unsafe { vq_flat_search(self.handle, &params, &mut n_results) };
        if results.is_null() || n_results == 0 {
            return vec![];
        }
        let vec: Vec<(u32, f32)> = unsafe {
            std::slice::from_raw_parts(results, n_results as usize)
                .iter()
                .map(|r| (r.id, r.distance))
                .collect()
        };
        unsafe { vq_results_free(results) };
        vec
    }

    pub fn ivf_batch_search(
        &self,
        queries: &[f32],
        n_queries: usize,
        d: usize,
        k: usize,
        nprobe: usize,
        refine_factor: usize,
        use_sq8: bool,
    ) -> Vec<Vec<(u32, f32)>> {
        let mut n_results_ptr: *mut c_int = std::ptr::null_mut();
        let results = unsafe {
            vq_ivf_batch_search(
                self.handle,
                queries.as_ptr(),
                n_queries as c_int,
                d as c_int,
                k as c_int,
                nprobe as c_int,
                refine_factor as c_int,
                use_sq8 as c_int,
                &mut n_results_ptr,
            )
        };

        if results.is_null() || n_results_ptr.is_null() {
            return vec![];
        }

        let mut all_results = Vec::with_capacity(n_queries);
        let mut offset = 0usize;
        unsafe {
            for q in 0..n_queries {
                let n = *n_results_ptr.add(q) as usize;
                let mut query_results = Vec::with_capacity(n);
                for i in 0..n {
                    let r = &*results.add(offset + i);
                    query_results.push((r.id, r.distance));
                }
                all_results.push(query_results);
                offset += n;
            }
            vq_results_free(results);
            vq_n_results_free(n_results_ptr);
        }
        all_results
    }
}

impl Drop for VectorEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { vq_engine_close(self.handle) };
        }
    }
}

unsafe impl Send for VectorEngine {}
unsafe impl Sync for VectorEngine {}
