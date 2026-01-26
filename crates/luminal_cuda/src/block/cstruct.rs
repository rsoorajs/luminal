use itertools::Itertools;
use luminal::{prelude::FxHashMap, shape::Expression};

#[derive(Debug, PartialEq, Eq)]
enum CStructTemplateType {
    Float,
    FloatArr(usize),
    Int,
    IntArr(usize),
    Long,
    LongArr(usize),
    Bool,
    BoolArr(usize),
    Ptr,
    PtrArr(usize),
    Bytes(usize),
}

#[derive(Debug)]
pub struct CStructData<'a> {
    buf: Vec<u8>,
    max_align: usize,
    struct_types: Vec<(String, CStructTemplateType)>,
    expressions: Option<&'a FxHashMap<Expression, i32>>,
    pub(crate) recorded_expressions: Vec<Expression>,
}

impl<'a> CStructData<'a> {
    pub fn new(expressions: Option<&'a FxHashMap<Expression, i32>>) -> Self {
        Self {
            max_align: 1,
            struct_types: vec![],
            buf: vec![],
            expressions,
            recorded_expressions: vec![],
        }
    }

    fn align_to(&mut self, align: usize) {
        self.max_align = self.max_align.max(align);

        let len = self.buf.len();
        let rem = len % align;
        if rem != 0 {
            let pad = align - rem;
            self.buf.extend(std::iter::repeat_n(0u8, pad));
        }
    }

    pub fn int(mut self, name: impl ToString, v: i32) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::Int));
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn int_arr(mut self, name: impl ToString, vs: &[i32]) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::IntArr(vs.len())));
        self.align_to(4);
        for &v in vs {
            self.buf.extend_from_slice(&v.to_ne_bytes());
        }
        self
    }

    pub fn expr(mut self, name: impl ToString, v: impl Into<Expression>) -> Self {
        if let Some(expressions) = self.expressions {
            self.struct_types
                .push((name.to_string(), CStructTemplateType::Int));
            let v = expressions[&v.into()];
            self.align_to(4);
            self.buf.extend_from_slice(&v.to_ne_bytes());
        } else {
            self.recorded_expressions.push(v.into());
        }
        self
    }

    pub fn expr_arr(mut self, name: impl ToString, vs: &[Expression]) -> Self {
        if let Some(expressions) = self.expressions {
            self.struct_types
                .push((name.to_string(), CStructTemplateType::IntArr(vs.len())));
            self.align_to(4);
            for &v in vs {
                let v = expressions[&v.into()];
                self.buf.extend_from_slice(&v.to_ne_bytes());
            }
        } else {
            self.recorded_expressions.extend(vs.iter().copied());
        }
        self
    }

    pub fn long(mut self, name: impl ToString, v: i64) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::Long));
        self.align_to(8);
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn float(mut self, name: impl ToString, v: f32) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::Float));
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn float_arr(mut self, name: impl ToString, vs: &[f32]) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::FloatArr(vs.len())));
        self.align_to(4);
        for &v in vs {
            self.buf.extend_from_slice(&v.to_ne_bytes());
        }
        self
    }

    pub fn bool(mut self, name: impl ToString, v: bool) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::Bool));
        self.align_to(1);
        self.buf.push(if v { 1 } else { 0 });
        self
    }

    pub fn ptr_const_f32(mut self, name: impl ToString, p: *const f32) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::Ptr));
        let ptr_size = std::mem::size_of::<usize>(); // usually 8
        let ptr_align = ptr_size;
        self.align_to(ptr_align);

        let addr = p as usize;
        let bytes = addr.to_ne_bytes();

        self.buf.extend_from_slice(&bytes[..ptr_size]);
        self
    }

    pub fn ptr_mut_f32(self, name: impl ToString, p: *mut f32) -> Self {
        self.ptr_const_f32(name, p as *const f32)
    }

    pub fn ptr_const_f32_arr(mut self, name: impl ToString, p: &[*const f32]) -> Self {
        self.struct_types
            .push((name.to_string(), CStructTemplateType::PtrArr(p.len())));
        let ptr_size = std::mem::size_of::<usize>(); // usually 8
        let ptr_align = ptr_size;
        self.align_to(ptr_align);

        for &p in p {
            let addr = p as usize;
            let bytes = addr.to_ne_bytes();
            self.buf.extend_from_slice(&bytes[..ptr_size]);
        }
        self
    }

    /// Returns the current size of the buffer after alignment for a pointer field.
    /// Useful for computing field offsets.
    pub fn current_size(&self) -> usize {
        let ptr_align = std::mem::size_of::<usize>();
        let len = self.buf.len();
        let rem = len % ptr_align;
        if rem != 0 {
            len + (ptr_align - rem)
        } else {
            len
        }
    }

    /// Pad the struct size to a multiple of max_align.
    pub fn finish_struct(mut self) -> Vec<u8> {
        assert!(
            self.expressions.is_some(),
            "Can only create cstruct bytes when expression map is provided!"
        );
        let align = self.max_align;
        if align > 1 {
            let len = self.buf.len();
            let rem = len % align;
            if rem != 0 {
                let pad = align - rem;
                self.buf.extend(std::iter::repeat_n(0u8, pad));
            }
        }
        self.buf
    }

    /// Insert a raw byte field (e.g., another struct).
    /// `align` must be the alignment of the nested struct.
    pub fn bytes(mut self, align: usize, data: &[u8]) -> Self {
        self.align_to(align);
        self.buf.extend_from_slice(data);
        self
    }
}

impl ToString for CStructData<'_> {
    fn to_string(&self) -> String {
        self.struct_types
            .iter()
            .map(|(name, ty)| match ty {
                CStructTemplateType::Bool => format!("bool {name};"),
                CStructTemplateType::BoolArr(l) => format!("bool {name}[{l}];"),
                CStructTemplateType::Float => format!("float {name};"),
                CStructTemplateType::FloatArr(l) => format!("float {name}[{l}];"),
                CStructTemplateType::Int => format!("int {name};"),
                CStructTemplateType::IntArr(l) => format!("int {name}[{l}];"),
                CStructTemplateType::Long => format!("long {name};"),
                CStructTemplateType::LongArr(l) => format!("long {name}[{l}];"),
                CStructTemplateType::Ptr => format!("float* {name};"),
                CStructTemplateType::PtrArr(l) => format!("float* {name}[{l}];"),
                CStructTemplateType::Bytes(l) => format!("char payload[{l}];"),
            })
            .join("\n")
    }
}
