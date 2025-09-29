(*
 * Copyright 2020, Data61, CSIRO (ABN 41 687 119 230)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *)

theory Hardware_H
imports
  MachineOps
  State_H
begin

context Arch begin arch_global_naming (H)

#INCLUDE_HASKELL SEL4/Machine/Hardware/RISCV64.hs Platform=Platform.RISCV64 CONTEXT RISCV64_H NOT plic_complete_claim getMemoryRegions getDeviceRegions getKernelDevices loadWord storeWord storeWordVM getActiveIRQ ackInterrupt maskInterrupt configureTimer resetTimer debugPrint getRestartPC setNextPC clearMemory clearMemoryVM initMemory freeMemory setHardwareASID wordFromPDE wordFromPTE VMFaultType HypFaultType VMPageSize pageBits pageBitsForSize toPAddr addrFromPPtr ptrFromPAddr sfence physBase paddrBase pptrBase pptrBaseOffset pptrTop pptrUserTop kernelELFBase kernelELFBaseOffset kernelELFPAddrBase addrFromKPPtr ptTranslationBits vmFaultTypeFSR read_stval setVSpaceRoot hwASIDFlush setIRQTrigger

end

arch_requalify_types (H)
  vmrights

context Arch begin arch_global_naming (H)

#INCLUDE_HASKELL SEL4/Machine/Hardware/RISCV64.hs CONTEXT RISCV64_H instanceproofs NOT plic_complete_claim HardwareASID VMFaultType VMPageSize VMPageEntry HypFaultType

#INCLUDE_HASKELL SEL4/Machine/Hardware/RISCV64.hs CONTEXT RISCV64_H ONLY wordFromPTE

(* Unlike on Arm architectures, maxIRQ comes from Platform definitions.
   We provide this abbreviation to match arch-split expectations. *)
abbreviation (input) maxIRQ :: irq where
  "maxIRQ \<equiv> Platform.RISCV64.maxIRQ"

end (* context RISCV64 *)

end
