// Copyright © 2026 Francesco Magazzù
// SPDX-License-Identifier: MIT

//! Sets the Maxwell operand reuse flags.
//!
//! SM50-SM6x have a small operand reuse cache in front of each GPR
//! operand collector port: when an instruction flags a source slot for
//! reuse and the next instruction issued by the same warp reads the
//! same register through the same slot, the read is served from the
//! cache instead of the register bank, saving a bank access and the
//! conflicts that come with it (see the maxas Control-Codes notes).
//!
//! The flag is a performance hint with two correctness constraints:
//!
//! - the cache is not invalidated by register writes, so the flag must
//!   not be set when the instruction itself rewrites the register the
//!   next instruction reads;
//! - the cache only survives back-to-back issue from the same warp, so
//!   the flag is useless (and not obviously harmless) combined with a
//!   yield — we skip yielding instructions, which is also why this
//!   pass runs after calc_instr_deps.
//!
//! Slots map onto the Ra/Rb/Rc encoding fields.  With every source in
//! a plain register the SM50 ALU encoders place srcs[i] in field i;
//! immediate/cbuf forms move sources across fields (e.g. FFMA with a
//! cbuf multiplicand encodes srcs[1] in Rc), so only all-register
//! pairs are flagged.

use crate::ir::*;

/// Leading source slots that land 1:1 on the Ra/Rb/Rc collector ports
/// in the all-register SM50 encoding of this op.  Sources past these
/// (ISetP's accumulator, IMnMx's min predicate, ...) are not GPR port
/// operands and don't affect the mapping.
fn reuse_gpr_slots(op: &Op) -> usize {
    match op {
        Op::FAdd(_)
        | Op::FMul(_)
        | Op::FMnMx(_)
        | Op::IAdd2(_)
        | Op::IMnMx(_)
        | Op::ISetP(_)
        | Op::Lop2(_) => 2,
        Op::FFma(_) => 3,
        _ => 0,
    }
}

/// The encoders keep the register form (and with it the srcs[i] ->
/// port i mapping) for both plain registers and RZ.
fn is_reg_form(src: &Src) -> bool {
    matches!(&src.src_ref, SrcRef::Reg(_) | SrcRef::Zero)
}

fn src_gpr(src: &Src) -> Option<RegRef> {
    match &src.src_ref {
        SrcRef::Reg(reg) if reg.file() == RegFile::GPR && reg.comps() == 1 => {
            Some(*reg)
        }
        _ => None,
    }
}

fn writes_gpr(instr: &Instr, reg: &RegRef) -> bool {
    instr.dsts().iter().any(|dst| match dst {
        Dst::Reg(d) => {
            d.file() == RegFile::GPR
                && reg.base_idx() >= d.base_idx()
                && reg.base_idx() < d.base_idx() + u32::from(d.comps())
        }
        _ => false,
    })
}

impl Shader<'_> {
    pub fn assign_reuse_flags(&mut self) {
        let sm = self.sm.sm();
        if !(50..70).contains(&sm) {
            return;
        }

        for f in &mut self.functions {
            for blk in f.blocks.iter_mut() {
                for i in 0..blk.instrs.len().saturating_sub(1) {
                    let (head, tail) = blk.instrs.split_at_mut(i + 1);
                    let a = &mut head[i];
                    let b = &tail[0];

                    if a.deps.yld {
                        continue;
                    }

                    let a_slots = reuse_gpr_slots(&a.op);
                    let b_slots = reuse_gpr_slots(&b.op);
                    if a_slots == 0 || b_slots == 0 {
                        continue;
                    }
                    if !a.op.srcs_as_slice()[..a_slots]
                        .iter()
                        .all(is_reg_form)
                        || !b.op.srcs_as_slice()[..b_slots]
                            .iter()
                            .all(is_reg_form)
                    {
                        continue;
                    }

                    for s in 0..a_slots.min(b_slots) {
                        let Some(ra) = src_gpr(&a.op.srcs_as_slice()[s])
                        else {
                            continue;
                        };
                        let Some(rb) = src_gpr(&b.op.srcs_as_slice()[s])
                        else {
                            continue;
                        };
                        if ra != rb || writes_gpr(a, &ra) {
                            continue;
                        }
                        a.deps.add_reuse(s.try_into().unwrap());
                    }
                }
            }
        }
    }
}
