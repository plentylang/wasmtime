//! Keep scalar floating-point recurrences in their loop-header registers.
//! SSE2 arithmetic overwrites its left operand.

use crate::FxHashSet;
use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::entity::{EntitySet, SecondaryMap};
use crate::ir::{BlockArg, Function, Inst, InstBuilder, Opcode, Value, types};
use alloc::vec::Vec;

enum Edit {
    Commute {
        product: Inst,
        previous: Value,
    },
    SaveBeforeSquare {
        square: Inst,
        original: Value,
        branch: Inst,
        successor: usize,
    },
}

/// Select destructive operands before lowering. Splitting the live range
/// with a block argument gives the allocator a copy it can merge with the
/// next iteration's unsquared parameter. This runs after phi simplification,
/// which would otherwise discard the single-predecessor argument.
pub(super) fn rewrite_recurrences(func: &Function, domtree: &DominatorTree) -> Option<Function> {
    let mut edits = Vec::new();
    for block in func.layout.blocks() {
        let branch = func.layout.last_inst(block).unwrap();
        let data = &func.dfg.insts[branch];
        if !matches!(data.opcode(), Opcode::Jump | Opcode::Brif) {
            continue;
        }

        let pool = &func.dfg.value_lists;
        for (successor, call) in data
            .branch_destination(&func.dfg.jump_tables, &func.dfg.exception_tables)
            .iter()
            .enumerate()
        {
            let header = call.block(pool);
            let params = func.dfg.block_params(header);
            if params.is_empty() || !domtree.dominates(header, block, &func.layout) {
                continue;
            }
            let mut edge_values = None;
            for (&previous, arg) in params.iter().zip(call.args(pool)) {
                let BlockArg::Value(next) = arg else { continue };
                if !matches!(func.dfg.value_type(next), types::F32 | types::F64) {
                    continue;
                }
                // The header redefines its parameters on every entry. A
                // single static use in a nested loop can run repeatedly
                // before the header is entered again.
                if let Some(product) = recurrence_product(func, next, previous)
                    && func.layout.inst_block(product) == Some(header)
                {
                    edits.push(Edit::Commute { product, previous });
                }
                if let Some((square, original)) = square_of_accumulator(func, next, previous)
                    && func.layout.inst_block(square) == Some(header)
                    && edge_values
                        .get_or_insert_with(|| {
                            call.args(pool)
                                .filter_map(|arg| match arg {
                                    BlockArg::Value(value) => Some(value),
                                    _ => None,
                                })
                                .collect::<FxHashSet<_>>()
                        })
                        .contains(&original)
                {
                    edits.push(Edit::SaveBeforeSquare {
                        square,
                        original,
                        branch,
                        successor,
                    });
                }
            }
        }
    }

    if edits.is_empty() {
        return None;
    }

    // Count actual IR uses, including branch arguments. A preserved value can
    // only be overwritten when its square and one backedge are its sole users.
    // Likewise prefer a right operand only when this multiply consumes it.
    let mut uses = SecondaryMap::<Value, u8>::new();
    for inst in func
        .layout
        .blocks()
        .flat_map(|block| func.layout.block_insts(block))
    {
        for arg in func.dfg.inst_values(inst) {
            uses[arg] = uses[arg].saturating_add(1);
        }
    }
    let mut commuted = EntitySet::new();
    let mut rewritten = None;
    let mut split_blocks = EntitySet::new();
    for edit in edits {
        match edit {
            Edit::Commute { product, previous }
                if uses[previous] == 1 && commuted.insert(product) =>
            {
                let rewritten = rewritten.get_or_insert_with(|| func.clone());
                rewritten.dfg.inst_args_mut(product).swap(0, 1);
            }
            Edit::SaveBeforeSquare {
                square,
                original,
                branch,
                successor,
            } if uses[original] == 3 => {
                let block = func.layout.inst_block(square).unwrap();
                // A failed search stops at the next instruction. A successful
                // search reserves the block, bounding the total scan length.
                if !split_blocks.contains(block)
                    && let Some(before) = square_insertion_point(func, square)
                {
                    split_blocks.insert(block);
                    split_before_square(
                        rewritten.get_or_insert_with(|| func.clone()),
                        square,
                        before,
                        original,
                        branch,
                        successor,
                    );
                }
            }
            _ => {}
        }
    }
    rewritten
}

fn recurrence_product(func: &Function, next: Value, previous: Value) -> Option<Inst> {
    let update = func.dfg.value_def(next).inst()?;
    let product = match func.dfg.insts[update].opcode() {
        Opcode::Fmul => update,
        Opcode::Fadd | Opcode::Fsub => func.dfg.value_def(func.dfg.inst_args(update)[0]).inst()?,
        _ => return None,
    };
    // CLIF leaves the payload of noncanonical arithmetic NaNs unspecified.
    (func.dfg.insts[product].opcode() == Opcode::Fmul
        && func.dfg.inst_args(product)[1] == previous
        && func.dfg.inst_args(product)[0] != previous)
        .then_some(product)
}

/// The incoming square is the destructive left operand of the update.
fn square_of_accumulator(func: &Function, next: Value, previous: Value) -> Option<(Inst, Value)> {
    let square = func.dfg.value_def(next).inst()?;
    if func.dfg.insts[square].opcode() != Opcode::Fmul {
        return None;
    }
    let &[original, rhs] = func.dfg.inst_args(square) else {
        unreachable!("fmul is binary");
    };
    let update = func.dfg.value_def(original).inst()?;
    if original != rhs || !matches!(func.dfg.insts[update].opcode(), Opcode::Fadd | Opcode::Fsub) {
        return None;
    }
    let lhs = func.dfg.inst_args(update)[0];
    (lhs == previous
        || func.dfg.value_def(lhs).inst().is_some_and(|inst| {
            func.dfg.insts[inst].opcode() == Opcode::Fsub && func.dfg.inst_args(inst)[0] == previous
        }))
    .then_some((square, original))
}

/// Sink the square past independent scalar arithmetic so those instructions
/// can finish using the old unsquared value before the preservation copy.
fn square_insertion_point(func: &Function, square: Inst) -> Option<Inst> {
    let result = func.dfg.first_result(square);
    let next = func.layout.next_inst(square)?;
    let mut before = next;
    while matches!(
        func.dfg.insts[before].opcode(),
        Opcode::Fadd | Opcode::Fsub | Opcode::Fmul
    ) && !func.dfg.inst_args(before).contains(&result)
    {
        before = func.layout.next_inst(before)?;
    }
    (before != next).then_some(before)
}

fn split_before_square(
    func: &mut Function,
    square: Inst,
    before: Inst,
    original: Value,
    branch: Inst,
    successor: usize,
) {
    let block = func.layout.inst_block(before).unwrap();
    let tail = func.dfg.make_block();
    let ty = func.dfg.value_type(original);
    let saved = func.dfg.append_block_param(tail, ty);
    func.layout.split_block(tail, before);
    if func.layout.is_cold(block) {
        func.layout.set_cold(tail);
    }
    func.layout.remove_inst(square);
    func.layout.insert_inst(square, before);
    let loc = func.srcloc(square);
    let mut cursor = FuncCursor::new(func).with_srcloc(loc);
    cursor.goto_bottom(block);
    cursor.ins().jump(tail, &[BlockArg::Value(original)]);

    let dfg = &mut cursor.func.dfg;
    let call = &mut dfg.insts[branch]
        .branch_destination_mut(&mut dfg.jump_tables, &mut dfg.exception_tables)[successor];
    call.update_args(&mut dfg.value_lists, |arg| {
        if arg == BlockArg::Value(original) {
            BlockArg::Value(saved)
        } else {
            arg
        }
    });
}
