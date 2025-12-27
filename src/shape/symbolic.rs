use generational_box::{AnyStorage, GenerationalBox, Owner, SyncStorage};
use rustc_hash::FxHashMap;
use serde::{Serialize, Serializer};
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign,
        Neg, Rem, RemAssign, Sub, SubAssign,
    },
    sync::OnceLock,
};

use crate::{egglog_utils, serialized_egraph::SerializedEGraph};
use egglog::{prelude::RustSpan, var};
use egglog_ast::span::Span;
use egraph_serialize::{ClassId, NodeId};

type ExprBox = GenerationalBox<Vec<Term>, SyncStorage>;

static EXPR_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();

pub fn expression_owner() -> &'static Owner<SyncStorage> {
    EXPR_OWNER.get_or_init(SyncStorage::owner)
}

#[derive(Copy, Clone)]
pub struct Expression {
    pub terms: ExprBox,
}

impl Serialize for Expression {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Access the Vec<Term> inside the GenerationalBox and serialize it
        self.terms.read().serialize(serializer)
    }
}

impl Expression {
    pub fn new(terms: Vec<Term>) -> Self {
        Self {
            terms: expression_owner().insert(terms),
        }
    }

    pub fn is_acc(&self) -> bool {
        self.terms.read().iter().any(|i| matches!(i, Term::Acc(_)))
    }

    pub fn is_dynamic(&self) -> bool {
        self.terms.read().iter().any(|i| {
            if let Term::Var(v) = i {
                *v != 'z'
            } else {
                false
            }
        })
    }

    pub fn dyn_vars(&self) -> Vec<char> {
        self.terms
            .read()
            .iter()
            .filter_map(|i| {
                if let Term::Var(v) = i {
                    if *v != 'z' { Some(*v) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.terms.read().hash(state);
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::new(vec![])
    }
}

/// A single term of a symbolic expression such as a variable, number or operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Term {
    Num(i32),
    Var(char),
    Add,
    Sub,
    Mul,
    Div,
    CeilDiv,
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
    Acc(char),
}

impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Num(n) => write!(f, "{n}"),
            Term::Var(c) => write!(f, "{c}"),
            Term::Add => write!(f, "+"),
            Term::Sub => write!(f, "-"),
            Term::Mul => write!(f, "*"),
            Term::Div => write!(f, "/"),
            Term::Mod => write!(f, "%"),
            Term::Min => write!(f, "min"),
            Term::CeilDiv => write!(f, "^/"),
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
            Term::Acc(s) => write!(f, "{s}"),
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self::Num(0)
    }
}

impl Term {
    pub fn as_op(self) -> Option<fn(i64, i64) -> Option<i64>> {
        match self {
            Term::Add => Some(|a, b| a.checked_add(b)),
            Term::Sub => Some(|a, b| a.checked_sub(b)),
            Term::Mul => Some(|a, b| a.checked_mul(b)),
            Term::Div => Some(|a, b| a.checked_div(b)),
            Term::Mod => Some(|a, b| a.checked_rem(b)),
            Term::Max => Some(|a, b| Some(a.max(b))),
            Term::Min => Some(|a, b| Some(a.min(b))),
            Term::And => Some(|a, b| Some((a != 0 && b != 0) as i64)),
            Term::Or => Some(|a, b| Some((a != 0 || b != 0) as i64)),
            Term::Gte => Some(|a, b| Some((a >= b) as i64)),
            Term::Lt => Some(|a, b| Some((a < b) as i64)),
            Term::CeilDiv => Some(|a, b| Some(if a % b != 0 { a / b + 1 } else { a / b })),
            _ => None,
        }
    }
    pub fn as_float_op(self) -> Option<fn(f64, f64) -> f64> {
        match self {
            Term::Add => Some(|a, b| a + b),
            Term::Sub => Some(|a, b| a - b),
            Term::Mul => Some(|a, b| a * b),
            Term::Div => Some(|a, b| a / b),
            Term::Mod => Some(|a, b| a % b),
            Term::Max => Some(|a, b| a.max(b)),
            Term::Min => Some(|a, b| a.min(b)),
            Term::And => Some(|a, b| (a.abs() > 1e-4 && b.abs() > 1e-4) as i32 as f64),
            Term::Or => Some(|a, b| (a.abs() > 1e-4 || b.abs() > 1e-4) as i32 as f64),
            Term::Gte => Some(|a, b| (a >= b) as i32 as f64),
            Term::Lt => Some(|a, b| (a < b) as i32 as f64),
            Term::CeilDiv => Some(|a, b| (a / b).ceil()),
            _ => None,
        }
    }
    pub fn to_egglog(self) -> String {
        match self {
            Term::Add => "MAdd",
            Term::Sub => "MSub",
            Term::Mul => "MMul",
            Term::Div => "MDiv",
            Term::Mod => "MMod",
            Term::Max => "MMax",
            Term::Min => "MMin",
            Term::CeilDiv => "MCeilDiv",
            Term::Gte => "MGte",
            Term::Lt => "MLt",
            _ => panic!("egglog doesn't implement {self:?}"),
        }
        .to_string()
    }
}

impl<T> PartialEq<T> for Expression
where
    for<'a> &'a T: Into<Expression>,
{
    fn eq(&self, other: &T) -> bool {
        *self.terms.read() == *other.into().terms.read()
    }
}

impl From<&Expression> for Expression {
    fn from(value: &Expression) -> Self {
        *value
    }
}

impl Eq for Expression {}

impl Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => c.to_string(),
                Term::Acc(c) => format!("Acc({c})"),
                Term::Max => format!(
                    "max({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        write!(f, "{}", symbols.pop().unwrap_or_default())
    }
}

impl Expression {
    pub fn to_egglog(&self) -> String {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => format!("(MNum {n})"),
                Term::Var(c) => format!("(MVar \"{c}\")"),
                Term::Acc(s) => format!("(MAccum \"{s}\")"),
                Term::Max => format!(
                    "(MMax {} {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "(MMin {} {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({} {} {})",
                    term.to_egglog(),
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        symbols.pop().unwrap_or_default()
    }

    pub fn to_kernel(&self) -> String {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => format!("{}const_{c}", if *c == 'z' { "" } else { "*" }),
                Term::Acc(_) => unreachable!(),
                Term::Max => format!(
                    "max((int){}, (int){})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min((int){}, (int){})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Lt => format!(
                    "(int)({} < {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Gte => format!(
                    "(int)({} >= {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::CeilDiv => {
                    let a = symbols.pop().unwrap();
                    let b = symbols.pop().unwrap();
                    format!("(({a} + {b} - 1) / {b})")
                }
                Term::Div => format!("({} / {})", symbols.pop().unwrap(), symbols.pop().unwrap()),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        symbols.pop().unwrap_or_default()
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Expression {
    /// Simplify the expression to its minimal terms
    pub fn simplify(self) -> Self {
        if self.terms.read().len() == 1 {
            return self;
        }
        egglog_simplify(self, false)
    }

    /// Simplify the expression to its minimal terms, using a cache to retrieve / store the simplification
    #[allow(clippy::mutable_key_type)]
    pub fn simplify_cache(self, cache: &mut FxHashMap<Expression, Expression>) -> Self {
        if let Some(s) = cache.get(&self) {
            *s
        } else {
            let simplified = self.simplify();
            cache.insert(self, simplified);
            simplified
        }
    }

    pub fn as_num(&self) -> Option<i32> {
        if let Term::Num(n) = self.terms.read()[0] {
            if self.terms.read().len() == 1 {
                return Some(n);
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.terms.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Minimum
    pub fn min(self, rhs: impl Into<Self>) -> Self {
        let rhs = rhs.into();
        if rhs == self || rhs == i32::MAX {
            return self;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.min(b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Min);
        Expression::new(terms)
    }

    /// Maximum
    pub fn max<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self || self == i32::MAX {
            return self;
        }
        if rhs == i32::MAX {
            return rhs;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.max(b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Max);
        Expression::new(terms)
    }

    /// Greater than or equals
    pub fn gte<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self {
            return true.into();
        }
        if rhs == i32::MAX {
            return false.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a >= b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Gte);
        Expression::new(terms)
    }

    /// Ceil Division
    pub fn ceil_div<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::CeilDiv);
        Expression::new(terms)
    }

    /// Less than
    pub fn lt<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self {
            return false.into();
        }
        if let Term::Num(n) = rhs.terms.read()[0] {
            if self.terms.read()[self.terms.read().len() - 1] == Term::Mod
                && self.terms.read()[0] == Term::Num(n)
            {
                return true.into();
            }
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a < b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Lt);
        Expression::new(terms)
    }

    /// Substitute an expression for a variable
    pub fn substitute(self, var: char, expr: impl Into<Expression>) -> Self {
        let mut new_terms = vec![];
        let t = expr.into().terms.read();
        for term in self.terms.read().iter() {
            match term {
                Term::Var(c) if *c == var => {
                    for t in t.iter() {
                        new_terms.push(*t);
                    }
                }
                _ => {
                    new_terms.push(*term);
                }
            }
        }
        Expression::new(new_terms)
    }
}

impl Expression {
    /// Evaluate the expression with no variables. Returns Some(value) if no variables are required, otherwise returns None.
    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&FxHashMap::default())
    }
    /// Evaluate the expression with one value for all variables.
    pub fn exec_single_var(&self, value: usize) -> usize {
        let mut stack = Vec::new();
        self.exec_single_var_stack(value, &mut stack)
    }
    /// Evaluate the expression with one value for all variables. Uses a provided stack
    pub fn exec_single_var_stack(&self, value: usize, stack: &mut Vec<i64>) -> usize {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Acc(_) => stack.push(1),
                Term::Var(_) => stack.push(value as i64),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
                }
            }
        }
        stack.pop().unwrap() as usize
    }
    /// Evaluate the expression given variables.
    pub fn exec(&self, variables: &FxHashMap<char, usize>) -> Option<usize> {
        self.exec_stack(variables, &mut Vec::new())
    }
    /// Evaluate the expression given variables. This function requires a stack to be given for use as storage
    pub fn exec_stack(
        &self,
        variables: &FxHashMap<char, usize>,
        stack: &mut Vec<i64>,
    ) -> Option<usize> {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Acc(_) => stack.push(1),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as i64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
                }
            }
        }
        stack.pop().map(|i| i as usize)
    }
    /// Evaluate the expression given variables.
    pub fn exec_float(&self, variables: &FxHashMap<char, usize>) -> Option<f64> {
        self.exec_stack_float(variables, &mut Vec::new())
    }
    /// Evaluate the expression given variables. This function requires a stack to be given for use as storage
    pub fn exec_stack_float(
        &self,
        variables: &FxHashMap<char, usize>,
        stack: &mut Vec<f64>,
    ) -> Option<f64> {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as f64),
                Term::Acc(_) => stack.push(1.0),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as f64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_float_op().unwrap()(a, b));
                }
            }
        }
        stack.pop()
    }
    /// Retrieve all symbols in the expression.
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .read()
            .iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    /// Check if the '-' variable exists in the expression.
    pub fn is_unknown(&self) -> bool {
        self.terms
            .read()
            .iter()
            .any(|t| matches!(t, Term::Var('-')))
    }

    pub fn resolve_vars(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for term in self.terms.write().iter_mut() {
            if let Term::Var(v) = *term
                && let Some(val) = dyn_map.get(&v)
            {
                *term = Term::Num(*val as i32);
            }
        }
    }
}

impl From<Term> for Expression {
    fn from(value: Term) -> Self {
        Expression::new(vec![value])
    }
}

impl From<char> for Expression {
    fn from(value: char) -> Self {
        Expression::new(vec![Term::Var(value)])
    }
}

impl From<&char> for Expression {
    fn from(value: &char) -> Self {
        Expression::new(vec![Term::Var(*value)])
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&usize> for Expression {
    fn from(value: &usize) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl From<i32> for Expression {
    fn from(value: i32) -> Self {
        Expression::new(vec![Term::Num(value)])
    }
}

impl From<&i32> for Expression {
    fn from(value: &i32) -> Self {
        Expression::new(vec![Term::Num(*value)])
    }
}

impl From<bool> for Expression {
    fn from(value: bool) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&bool> for Expression {
    fn from(value: &bool) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl Add<Expression> for usize {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for usize {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for usize {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for usize {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for usize {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for usize {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for usize {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl Add<Expression> for i32 {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for i32 {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for i32 {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for i32 {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for i32 {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for i32 {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for i32 {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl Neg for Expression {
    type Output = Expression;
    fn neg(self) -> Self::Output {
        self * -1
    }
}

impl<E: Into<Expression>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 {
            return self;
        }
        if self == 0 {
            return rhs;
        }
        if self == rhs {
            return self * 2;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a + b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Add);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 {
            return self;
        }
        if self == rhs {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a - b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Sub);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            if let Some(c) = a.checked_mul(b) {
                return c.into();
            }
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mul);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == rhs {
            return 1.into();
        }
        if self == 0 {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            if a % b == 0 {
                if let Some(c) = a.checked_div(b) {
                    return c.into();
                }
            }
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Div);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 || rhs == self {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a % b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mod);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 && b != 0).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::And);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitOr<E> for Expression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 || self == 1 {
            return 1.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 || b != 0).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Or);
        Expression::new(terms)
    }
}

impl std::iter::Product for Expression {
    fn product<I: Iterator<Item = Expression>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p *= n;
        }
        p
    }
}

impl std::iter::Sum for Expression {
    fn sum<I: Iterator<Item = Expression>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p += n;
        }
        p
    }
}

impl<E: Into<Expression>> AddAssign<E> for Expression {
    fn add_assign(&mut self, rhs: E) {
        *self = *self + rhs;
    }
}

impl<E: Into<Expression>> SubAssign<E> for Expression {
    fn sub_assign(&mut self, rhs: E) {
        *self = *self - rhs;
    }
}

impl<E: Into<Expression>> MulAssign<E> for Expression {
    fn mul_assign(&mut self, rhs: E) {
        *self = *self * rhs;
    }
}

impl<E: Into<Expression>> DivAssign<E> for Expression {
    fn div_assign(&mut self, rhs: E) {
        *self = *self / rhs;
    }
}

impl<E: Into<Expression>> RemAssign<E> for Expression {
    fn rem_assign(&mut self, rhs: E) {
        *self = *self % rhs;
    }
}

impl<E: Into<Expression>> BitAndAssign<E> for Expression {
    fn bitand_assign(&mut self, rhs: E) {
        *self = *self & rhs;
    }
}

impl<E: Into<Expression>> BitOrAssign<E> for Expression {
    fn bitor_assign(&mut self, rhs: E) {
        *self = *self | rhs;
    }
}

fn extract_shortest<'a>(
    egraph: &'a SerializedEGraph,
    class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    let result = egraph.eclasses[class]
        .1
        .iter()
        .filter_map(|en| {
            if *seen.get(en).unwrap_or(&0) >= 4 || egraph.enodes[en].0 == "[...]" {
                return None;
            }
            if let Some(cached) = cache.get(en) {
                return cached.clone();
            }
            *seen.entry(en).or_insert(0) += 1;
            let out = if egraph.enodes[en].1.is_empty() {
                Some(vec![en])
            } else {
                egraph.enodes[en]
                    .1
                    .iter()
                    .try_fold(vec![en], |mut acc, ch| {
                        extract_shortest(egraph, ch, seen, cache).map(|p| {
                            acc.extend(p);
                            acc
                        })
                    })
            };
            *seen.get_mut(en).unwrap() -= 1;
            cache.insert(en, out.clone());
            out
        })
        .min_by_key(|p| p.len());
    result
}

fn build_expression(
    egraph: &SerializedEGraph,
    trajectory: &[&NodeId],
    current: &mut usize,
) -> Expression {
    let nid = trajectory[*current];
    let op = egraph.enodes[nid].0.as_str();
    match op {
        "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr" | "MGte"
        | "MLt" | "MFloorTo" | "MCeilDiv" => {
            *current += 1;
            let lhs = build_expression(egraph, trajectory, current);
            *current += 1;
            let rhs = build_expression(egraph, trajectory, current);
            match op {
                "MAdd" => lhs + rhs,
                "MSub" => lhs - rhs,
                "MMul" => lhs * rhs,
                "MDiv" => lhs / rhs,
                "MMod" => lhs % rhs,
                "MMin" => lhs.min(rhs),
                "MMax" => lhs.max(rhs),
                "MAnd" => lhs & rhs,
                "MOr" => lhs | rhs,
                "MGte" => lhs.gte(rhs),
                "MLt" => lhs.lt(rhs),
                "MCeilDiv" => lhs.ceil_div(rhs),
                "MFloorTo" => lhs / rhs * rhs,
                _ => unreachable!(),
            }
        }
        "MNum" | "MVar" | "MAccum" => {
            *current += 1;
            let child = build_expression(egraph, trajectory, current);
            if op == "MAccum" {
                if let Some(Term::Var(c)) = child.terms.read().first() {
                    Expression::new(vec![Term::Acc(*c)])
                } else {
                    child
                }
            } else {
                child
            }
        }
        "MIter" => Expression::from('z'),
        op if op.starts_with("Boxed(\"") => {
            let name = op.replace("Boxed(\"", "").replace("\")", "");
            Expression::from(name.chars().next().unwrap())
        }
        op => op
            .parse::<i32>()
            .map(Expression::from)
            .or_else(|_| op.replace('"', "").parse::<char>().map(Expression::from))
            .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
    }
}

fn extract_expression(egraph: &SerializedEGraph) -> Option<Expression> {
    let root = egraph.roots.first()?;
    let traj = extract_shortest(
        egraph,
        root,
        &mut FxHashMap::default(),
        &mut FxHashMap::default(),
    )?;
    Some(build_expression(egraph, &traj, &mut 0))
}

#[derive(Clone, Debug)]
enum ExprNode {
    Num(i32),
    Var(char),
    Acc(char),
    Op(Term, Box<ExprNode>, Box<ExprNode>),
}

impl ExprNode {
    fn from_terms(terms: &[Term]) -> Option<Self> {
        let mut stack: Vec<ExprNode> = Vec::new();
        for term in terms {
            match *term {
                Term::Num(n) => stack.push(ExprNode::Num(n)),
                Term::Var(c) => stack.push(ExprNode::Var(c)),
                Term::Acc(c) => stack.push(ExprNode::Acc(c)),
                op => {
                    let left = stack.pop()?;
                    let right = stack.pop()?;
                    stack.push(ExprNode::Op(op, Box::new(left), Box::new(right)));
                }
            }
        }
        stack.pop()
    }

    fn to_terms(&self, out: &mut Vec<Term>) {
        match self {
            ExprNode::Num(n) => out.push(Term::Num(*n)),
            ExprNode::Var(c) => out.push(Term::Var(*c)),
            ExprNode::Acc(c) => out.push(Term::Acc(*c)),
            ExprNode::Op(op, left, right) => {
                right.to_terms(out);
                left.to_terms(out);
                out.push(*op);
            }
        }
    }

    fn simplify_mul_div_constants(self) -> Self {
        match self {
            ExprNode::Op(term, left, right) => {
                let left = left.simplify_mul_div_constants();
                let right = right.simplify_mul_div_constants();
                if term == Term::Mul {
                    if let (ExprNode::Op(Term::Div, inner_left, inner_right), ExprNode::Num(c)) =
                        (&left, &right)
                    {
                        if let ExprNode::Num(b) = **inner_right {
                            if *c != 0 && b % *c == 0 && !inner_left.contains_var('z') {
                                return ExprNode::Op(
                                    Term::Div,
                                    inner_left.clone(),
                                    Box::new(ExprNode::Num(b / *c)),
                                );
                            }
                        }
                    }

                    if let (ExprNode::Num(c), ExprNode::Op(Term::Div, inner_left, inner_right)) =
                        (&left, &right)
                    {
                        if let ExprNode::Num(b) = **inner_right {
                            if *c != 0 && b % *c == 0 && !inner_left.contains_var('z') {
                                return ExprNode::Op(
                                    Term::Div,
                                    inner_left.clone(),
                                    Box::new(ExprNode::Num(b / *c)),
                                );
                            }
                        }
                    }
                }
                ExprNode::Op(term, Box::new(left), Box::new(right))
            }
            other => other,
        }
    }

    fn contains_var(&self, var: char) -> bool {
        match self {
            ExprNode::Var(c) => *c == var,
            ExprNode::Acc(_) | ExprNode::Num(_) => false,
            ExprNode::Op(_, left, right) => left.contains_var(var) || right.contains_var(var),
        }
    }
}

fn simplify_mul_div_constants(expr: Expression) -> Expression {
    let Some(node) = ExprNode::from_terms(&expr.terms.read()) else {
        return expr;
    };
    let simplified = node.simplify_mul_div_constants();
    let mut terms = Vec::new();
    simplified.to_terms(&mut terms);
    Expression::new(terms)
}

fn egglog_simplify(e: Expression, lower_bound_zero: bool) -> Expression {
    let expr = e.to_egglog();
    let mut program = String::new();
    program.push_str(egglog_utils::BASE);
    program.push('\n');
    program.push_str("(ruleset cleanup)\n");
    program.push_str(egglog_utils::BASE_CLEANUP);
    program.push('\n');
    program.push_str(
        r#"
(rewrite (MDiv (MDiv a b) c) (MDiv a (MMul b c)) :ruleset expr)
(rewrite (MAdd (MDiv a b) c) (MDiv (MAdd a (MMul c b)) b) :ruleset expr)
(rewrite (MAdd a (MSub b a)) b :ruleset expr)
(rewrite (MAdd (MSub b a) a) b :ruleset expr)
(rewrite (MSub a a) (MNum 0) :ruleset expr)
(rewrite
    (MAdd (MSub a (MNum ?b)) (MNum ?c))
    (MSub a (MNum (- ?b ?c)))
    :ruleset expr
)
(rewrite
    (MAdd (MNum ?c) (MSub a (MNum ?b)))
    (MSub a (MNum (- ?b ?c)))
    :ruleset expr
)
(rewrite
    (MSub (MAdd a (MNum ?b)) (MNum ?c))
    (MAdd a (MNum (- ?b ?c)))
    :ruleset expr
)
(rewrite
    (MSub (MSub a (MNum ?b)) (MNum ?c))
    (MSub a (MNum (+ ?b ?c)))
    :ruleset expr
)
(rewrite (MAdd (MMul a b) (MMul a c)) (MMul a (MAdd b c)) :ruleset expr)
(rewrite (MAdd a a) (MMul (MNum 2) a) :ruleset expr)
"#,
    );
    if lower_bound_zero {
        program.push_str("(rewrite (MMax a (MNum 0)) a :ruleset expr)\n");
    }
    program.push_str(&format!("(let expr_root {expr})\n"));
    program.push_str(egglog_utils::RUN_SCHEDULE);

    let mut egraph = egglog::EGraph::default();
    let commands = egraph
        .parser
        .get_program_from_string(None, &program)
        .expect("failed to parse egglog program");
    egraph
        .run_program(commands)
        .expect("failed to run egglog program");
    let (sort, value) = egraph
        .eval_expr(&var!("expr_root"))
        .expect("failed to evaluate egglog expression");
    let serialized = SerializedEGraph::new(&egraph, vec![(sort, value)]);
    extract_expression(&serialized)
        .map(simplify_mul_div_constants)
        .unwrap_or(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = Expression::from('x') + (256 - (Expression::from('x') % 256));
        assert_eq!(
            n.simplify()
                .exec(&[('x', 767)].into_iter().collect())
                .unwrap(),
            768
        );
    }

    #[test]
    fn test_minimizations() {
        let expr = ((Expression::from('a') * 1) + 0) / 1 + (1 - 1);
        let reduced_expr = expr.simplify();
        assert_eq!(reduced_expr, 'a');
    }

    #[test]
    fn test_substitution() {
        let main = Expression::from('x') - 255;
        let sub = Expression::from('x') / 2;
        let new = main.substitute('x', sub).simplify();
        assert_eq!(new.len(), 5);
    }

    #[test]
    fn test_group_terms() {
        let s = Expression::from('s');
        let expr = (s * ((s - 4) + 1)) + (((s + 1) * ((s - 4) + 1)) - (s * ((s - 4) + 1)));
        assert_eq!(expr.simplify().len(), 7);
    }

    #[test]
    fn test_simple_div() {
        let w = Expression::from('w');
        let s = ((((w + 3) / 2) + 2) / 2).simplify();
        assert_eq!(s, (w + 7) / 4);
    }

    #[test]
    fn test_other() {
        let z = Expression::from('z');
        let w = Expression::from('w');
        let h = Expression::from('h');
        let o = (z
            / ((-5 + (((((-5 + ((((((w + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
                * (-5 + (((9 + (4 * (-5 + ((((((153 + h) / 2) / 2) / 2) / 2) / 2)))) / 2) / 2))))
            % 64;
        let x = o.simplify();
        assert_eq!(x.len(), 23); // Should be 21 if we can re-enable mul-div-associative-rev
    }

    #[test]
    fn test_final() {
        let z = Expression::from('z');
        let w = Expression::from('w');
        let h = Expression::from('h');
        let x = z % (((((153 + h) / 8) + -31) * ((((w + 153) / 8) + -31) / 16)) * 64);
        let x = x.simplify();
        assert_eq!(x.len(), 15); // Should be 11 if we can re-enable mul-div-associative-rev
    }
}
