// SPDX-License-Identifier: MIT

//! Expand the SM120 ray-query meta-op into the known-working hardware sequence.

use crate::ir::*;

/*
 * The order of these descriptors is the operand order of
 * nir_ttu_op_bundle_nv and OpTtuOpBundle.  Slot values name TTU-internal
 * transfer locations; they are not register numbers or memory offsets.
 */
struct StoreDesc {
    slot: u8,
    src: usize,
}

const STORES: [StoreDesc; 8] = [
    StoreDesc { slot: 0x00, src: 0 }, // primary control
    StoreDesc { slot: 0x04, src: 1 }, // ray origin and tmin
    StoreDesc { slot: 0x05, src: 2 }, // ray direction and tmax
    StoreDesc { slot: 0x08, src: 3 }, // secondary control
    StoreDesc { slot: 0x10, src: 4 }, // continuation group 0
    StoreDesc { slot: 0x18, src: 5 }, // continuation group 1
    StoreDesc { slot: 0x19, src: 6 }, // continuation group 2
    StoreDesc { slot: 0x20, src: 4 }, // root node
];

struct LoadDesc {
    slot: u8,
    close: bool,
}

const LOADS: [LoadDesc; 6] = [
    // Continuation groups 0, 1, and 2
    LoadDesc {
        slot: 0x38,
        close: false,
    },
    LoadDesc {
        slot: 0x3c,
        close: false,
    },
    LoadDesc {
        slot: 0x3d,
        close: false,
    },
    // Instance and SBT metadata
    LoadDesc {
        slot: 0x32,
        close: false,
    },
    // Metadata context
    LoadDesc {
        slot: 0x31,
        close: false,
    },
    // Event payload and bundle close
    LoadDesc {
        slot: 0x30,
        close: true,
    },
];

fn src_to_pred(src: Src) -> Pred {
    assert!(src.src_ref.is_predicate());
    let (pred_ref, base_inv) = match src.src_ref {
        SrcRef::True => (PredRef::None, false),
        SrcRef::False => (PredRef::None, true),
        SrcRef::Reg(reg) => (PredRef::Reg(reg), false),
        _ => panic!("invalid TTU predicate"),
    };
    let modifier_inv = match src.src_mod {
        SrcMod::None => false,
        SrcMod::BNot => true,
        _ => panic!("invalid TTU predicate modifier"),
    };
    Pred {
        pred_ref,
        pred_inv: base_inv ^ modifier_inv,
    }
}

impl Shader<'_> {
    pub fn lower_ttu_op_bundle(&mut self) {
        /*
         * Expand one allocation-level bundle without changing its
         * already allocated operands:
         *
         *   emit TTUOPEN
         *   emit TTUMACROFUSE
         *   for i in 0..8:
         *       emit @predicate[i] TTUST input_slot[i],
         *                               input_pair[2*i],
         *                               input_pair[2*i + 1]
         *   emit TTUGO
         *   for i in 0..5:
         *       emit TTULD output_pair[2*i], output_pair[2*i + 1],
         *                  output_slot[i]
         *   emit TTULD.CLOSE output_pair[10], output_pair[11], output_slot[5]
         *
         * No generic instruction may be inserted inside this expansion.
         * Consequently it runs after optimization, legalization, register
         * allocation, and instruction scheduling.  Dependency assignment
         * runs immediately afterward to add ordinary scoreboard barriers
         * without replacing the packet's fixed issue delays.
         */
        self.map_instrs(|instr, _| {
            let Op::TtuOpBundle(bundle) = instr.op else {
                return [instr].into();
            };
            assert!(instr.pred.is_true());
            let (gpr_srcs, pred_srcs, dsts) = bundle.into_parts();

            let mut gpr_srcs = gpr_srcs.into_iter();
            let pairs: [(Src, Src); 7] = std::array::from_fn(|_| {
                (gpr_srcs.next().unwrap(), gpr_srcs.next().unwrap())
            });
            let predicates: [Pred; 8] = pred_srcs.map(src_to_pred);
            debug_assert!(gpr_srcs.next().is_none());

            fn make_instr(op: impl Into<Op>, delay: u8) -> Instr {
                let mut instr = Instr::new(op);
                instr.deps.set_delay(delay);
                instr
            }
            let mut out: Vec<Instr> = Vec::with_capacity(17);
            out.push(make_instr(OpTtuOpen {}, 6));
            out.push(make_instr(OpTtuMacroFuse {}, 4));
            for (desc, predicate) in STORES.iter().zip(predicates) {
                let (pair_a, pair_b) = pairs[desc.src].clone();
                let mut part = make_instr(
                    OpTtuStore {
                        pairs: [pair_a, pair_b],
                        slot: desc.slot,
                    },
                    1,
                );
                part.pred = predicate;
                out.push(part);
            }
            out.push(make_instr(OpTtuGo {}, 1));

            let mut dsts = dsts.into_iter();
            for desc in LOADS {
                out.push(make_instr(
                    OpTtuLoad {
                        pairs: [dsts.next().unwrap(), dsts.next().unwrap()],
                        slot: desc.slot,
                        close: desc.close,
                    },
                    if desc.close { 2 } else { 1 },
                ));
            }
            assert!(dsts.next().is_none());
            out.into()
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use compiler::cfg::CFGBuilder;
    use rustc_hash::FxBuildHasher;

    fn gpr_pair(base: u32) -> RegRef {
        assert!(base.is_multiple_of(2));
        RegRef::new(RegFile::GPR, base, 2)
    }

    pub(crate) fn test_shader<'a>(
        sm: &'a ShaderModelInfo,
        instrs: Vec<Instr>,
    ) -> Shader<'a> {
        let mut label_alloc = LabelAllocator::new();
        let block = BasicBlock {
            label: label_alloc.alloc(),
            uniform: true,
            instrs,
        };
        let mut cfg = CFGBuilder::<_, _, FxBuildHasher>::new();
        cfg.add_node(0, block);
        let f = Function {
            ssa_alloc: SSAValueAllocator::new(),
            phi_alloc: PhiAllocator::new(),
            blocks: cfg.as_cfg(true),
        };
        let info = ShaderInfo {
            max_warps_per_sm: 0,
            num_gprs: 0,
            num_control_barriers: 0,
            num_instrs: 0,
            num_static_cycles: 0,
            num_spills_to_mem: 0,
            num_fills_from_mem: 0,
            num_spills_to_reg: 0,
            num_fills_from_reg: 0,
            slm_size: 0,
            max_crs_depth: 0,
            uses_global_mem: true,
            writes_global_mem: true,
            uses_fp64: false,
            stage: ShaderStageInfo::Compute(ComputeShaderInfo {
                local_size: [32, 1, 1],
                smem_size: 0,
            }),
            io: ShaderIoInfo::None,
        };
        Shader {
            sm,
            info,
            functions: vec![f],
        }
    }

    pub(crate) fn allocated_bundle(
        last_predicate: Src,
        outputs: [u32; 12],
    ) -> OpTtuOpBundle {
        let mut gpr_srcs: [Src; 14] = std::array::from_fn(|_| true.into());
        for i in 0..7 {
            let base = u32::try_from(i).unwrap() * 4;
            gpr_srcs[i * 2] = SrcRef::Reg(gpr_pair(base)).into();
            gpr_srcs[i * 2 + 1] = SrcRef::Reg(gpr_pair(base + 2)).into();
        }
        let mut pred_srcs = std::array::from_fn(|_| true.into());
        pred_srcs[7] = last_predicate;
        let dsts = outputs.map(|reg| Dst::Reg(gpr_pair(reg)));
        OpTtuOpBundle::new(gpr_srcs, pred_srcs, dsts)
    }

    #[test]
    fn lowers_false_predicate_in_packet() {
        let sm = ShaderModelInfo::new(120, 0);
        let outputs =
            std::array::from_fn(|i| 32 + u32::try_from(i).unwrap() * 2);
        let bundle =
            Op::TtuOpBundle(Box::new(allocated_bundle(false.into(), outputs)));
        assert!(!bundle.is_uniform());
        let mut s = test_shader(&sm, vec![Instr::new(bundle)]);
        s.lower_ttu_op_bundle();

        let instrs = &s.functions[0].blocks[0].instrs;
        assert_eq!(instrs.len(), 17);
        assert!(matches!(
            &instrs[16].op,
            Op::TtuLoad(op) if op.close
        ));
        assert!(instrs[9].pred.is_false());
    }

    #[test]
    fn preserves_packet_scheduling() {
        let sm = ShaderModelInfo::new(120, 0);
        let outputs =
            std::array::from_fn(|i| 64 + u32::try_from(i).unwrap() * 2);
        let bundle =
            Op::TtuOpBundle(Box::new(allocated_bundle(true.into(), outputs)));
        let mut s = test_shader(&sm, vec![Instr::new(bundle)]);
        s.lower_ttu_op_bundle();
        s.calc_instr_deps();

        let instrs = &s.functions[0].blocks[0].instrs;
        assert_eq!(instrs[0].deps.delay, 6);
        assert_eq!(instrs[1].deps.delay, 4);
        assert!(instrs[2..16].iter().all(|instr| instr.deps.delay == 1));
        assert_eq!(instrs[16].deps.delay, 2);
        assert_eq!(instrs[10].deps.wt_bar_mask, 0);
    }
}
