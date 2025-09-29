(*
 * Copyright 2020, Data61, CSIRO (ABN 41 687 119 230)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *)

chapter "Architecture specific Retype definitions"

theory ArchRetype_A
imports
  ArchVSpaceAcc_A
  ArchInvocation_A
begin

context Arch begin arch_global_naming (A)

text \<open>
  This is a placeholder function. We may wish to extend the specification
  with explicitly tagging kernel data regions in memory.
\<close>
definition reserve_region :: "obj_ref \<Rightarrow> nat \<Rightarrow> bool \<Rightarrow> (unit,'z::state_ext) s_monad"
  where
  "reserve_region ptr byteLength is_kernel \<equiv> return ()"

text \<open>Initialise architecture-specific objects.\<close>

definition init_arch_objects ::
  "apiobject_type \<Rightarrow> bool \<Rightarrow> obj_ref \<Rightarrow> nat \<Rightarrow> nat \<Rightarrow> obj_ref list \<Rightarrow> (unit,'z::state_ext) s_monad"
  where
  "init_arch_objects new_type is_device ptr num_objects obj_sz refs \<equiv> return ()"

definition empty_context :: user_context
  where
  "empty_context \<equiv> UserContext (\<lambda>_. 0)"

definition init_arch_tcb :: arch_tcb
  where
  "init_arch_tcb \<equiv> \<lparr> tcb_context = empty_context \<rparr>"

end
end
