// Copyright 2020 Red Hat.
// SPDX-License-Identifier: MIT

use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::c_char;

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn c_string_to_string(cstr: *const c_char) -> String {
    if cstr.is_null() {
        return String::from("");
    }

    let res = unsafe { CStr::from_ptr(cstr).to_str() };
    assert!(res.is_ok());
    String::from(res.unwrap_or(""))
}

/// Simple wrapper around [`CStr::from_ptr`] to bind the lifetime to a slice containing the
/// nul-terminated string.
///
/// # Safety
///
/// Same as [`CStr::from_ptr`]
pub unsafe fn char_arr_to_cstr(c_str: &[c_char]) -> &CStr {
    // We make sure that at least the end of the slice is nul-terminated. We don't really care if
    // there is another nul within the slice.
    debug_assert_eq!(c_str.last().copied(), Some(b'\0' as c_char));
    unsafe { CStr::from_ptr(c_str.as_ptr()) }
}

pub trait CStringExt {
    fn push_cstr(&mut self, other: &CStr);
}

impl CStringExt for CString {
    fn push_cstr(&mut self, other: &CStr) {
        let mut buffer = mem::take(self).into_bytes();
        buffer.extend(other.to_bytes_with_nul());

        // SAFETY: The only 0 byte in buffer is at the end
        *self = unsafe { CString::from_vec_with_nul_unchecked(buffer) };
    }
}

pub trait Join<Separator> {
    type Output;

    fn join(&self, sep: Separator) -> Self::Output;
}

impl Join<&CStr> for [CString] {
    type Output = CString;

    fn join(&self, sep: &CStr) -> Self::Output {
        let Some((first, rest)) = self.split_first() else {
            return CString::default();
        };

        let size = self.iter().map(|v| v.count_bytes()).sum::<usize>()
            + sep.count_bytes() * (self.len() - 1)
            + 1;

        let mut buffer = Vec::with_capacity(size);
        buffer.extend(first.to_bytes());

        let sep = sep.to_bytes();
        for string in rest {
            buffer.extend_from_slice(sep);
            buffer.extend_from_slice(string.to_bytes());
        }

        // SAFETY: There are no interior 0 bytes in buffer
        unsafe { CString::from_vec_unchecked(buffer) }
    }
}

#[test]
fn test_push_cstr() {
    let mut owned = c"SomeString".to_owned();
    owned.push_cstr(c"MoreString");
    assert_eq!(owned, c"SomeStringMoreString".to_owned());
}

#[test]
fn test_join_c_strings() {
    assert_eq!(Join::join([].as_slice(), c";"), c"".to_owned());
    assert_eq!([c"test".to_owned()].join(c";"), c"test".to_owned());
    assert_eq!(
        [c"String".to_owned(), c"AnotherString".to_owned()].join(c";"),
        c"String;AnotherString".to_owned()
    );
    assert_eq!(
        [
            c"String".to_owned(),
            c"AnotherString".to_owned(),
            c"Testing".to_owned()
        ]
        .join(c"###"),
        c"String###AnotherString###Testing".to_owned(),
    );
}
