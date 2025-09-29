(*
 * Copyright 2020, Data61, CSIRO (ABN 41 687 119 230)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *)

chapter "Architecture-specific TCB functions"

theory ArchTcb_A
imports KHeap_A
begin

context Arch begin arch_global_naming (A)

definition sanitise_register :: "bool \<Rightarrow> register \<Rightarrow> machine_word \<Rightarrow> machine_word"
  where
  "sanitise_register t r v \<equiv> v"

definition arch_get_sanitise_register_info :: "obj_ref \<Rightarrow> (bool, 'a::state_ext) s_monad"
  where
  "arch_get_sanitise_register_info t \<equiv> return False"

definition arch_post_modify_registers :: "obj_ref \<Rightarrow> obj_ref \<Rightarrow> (unit, 'a::state_ext) s_monad"
  where
  "arch_post_modify_registers cur t \<equiv> return ()"

end
end
