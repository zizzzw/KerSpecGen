(*
 * Copyright 2014, General Dynamics C4 Systems
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *)

theory ArchStructures_H
imports
  "Lib.Lib"
  Types_H
  Hardware_H
begin

context Arch begin arch_global_naming (H)

#INCLUDE_SETTINGS keep_constructor=asidpool
#INCLUDE_SETTINGS keep_constructor=arch_tcb

#INCLUDE_HASKELL SEL4/Object/Structures/RISCV64.hs CONTEXT RISCV64_H decls_only
#INCLUDE_HASKELL SEL4/Object/Structures/RISCV64.hs CONTEXT RISCV64_H instanceproofs
#INCLUDE_HASKELL SEL4/Object/Structures/RISCV64.hs CONTEXT RISCV64_H bodies_only

datatype arch_kernel_object_type =
    PTET
  | ASIDPoolT

primrec
  archTypeOf :: "arch_kernel_object \<Rightarrow> arch_kernel_object_type"
where
  "archTypeOf (KOPTE e) = PTET"
| "archTypeOf (KOASIDPool e) = ASIDPoolT"

end
end
