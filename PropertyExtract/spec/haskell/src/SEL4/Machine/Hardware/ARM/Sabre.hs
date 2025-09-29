--
-- Copyright 2014, General Dynamics C4 Systems
--
-- SPDX-License-Identifier: GPL-2.0-only
--

{-# LANGUAGE EmptyDataDecls, ForeignFunctionInterface, GeneralizedNewtypeDeriving #-}

module SEL4.Machine.Hardware.ARM.Sabre where

import Prelude hiding (Word)
import SEL4.Machine.RegisterSet
import SEL4.Machine.Hardware.ARM.Callbacks
import SEL4.Machine.Hardware.GICInterface hiding (IRQ, maskInterrupt)
import qualified SEL4.Machine.Hardware.GICInterface as GIC
import qualified SEL4.Machine.Hardware.MPTimerInterface as MPT
import Foreign.Ptr
import Data.Bits

-- Following harded coded address pair are used in getKernelDevices
-- and will get mapped into kernel address space via mapKernelFrame
gicControllerBase = PAddr 0x00A00000
gicDistributorBase = PAddr 0x00A01000
l2ccBase = PAddr 0x00A02000
uartBase = PAddr 0x021E8000

uart = (uartBase, PPtr 0xfff01000)
l2cc = (l2ccBase, PPtr 0xfff03000)
gicController = (gicControllerBase, PPtr 0xfff04000)
gicDistributor = (gicDistributorBase, PPtr 0xfff05000)


gicInterfaceBase = gicControllerBase + 0x100
mptBase = gicControllerBase + 0x600

physBase :: PAddr
physBase = PAddr 0x10000000

pptrBase :: VPtr
pptrBase = VPtr 0xe0000000

pageColourBits :: Int
pageColourBits = 0 -- qemu has no cache

getMemoryRegions :: Ptr CallbackData -> IO [(PAddr, PAddr)]
getMemoryRegions _ = return [(physBase, physBase + (0x8 `shiftL` 24))]


userTimer = 0x020D4000

getDeviceRegions :: Ptr CallbackData -> IO [(PAddr, PAddr)]
getDeviceRegions _ = return devices
    where devices = [(userTimer,userTimer + (1 `shiftL` 12))]

type IRQ = GIC.IRQ

timerIRQ = GIC.IRQ 29

getKernelDevices :: Ptr CallbackData -> IO [(PAddr, PPtr Word)]
getKernelDevices _ = return devices
    where devices = [
            gicController, -- interrupt controller
            gicDistributor, -- interrupt controller
            uart
            ]

maskInterrupt :: Ptr CallbackData -> Bool -> IRQ -> IO ()
maskInterrupt env mask irq = do
     callGICApi (GicState { env = env, gicDistBase = gicDistributorBase, gicIFBase = gicInterfaceBase })
       (GIC.maskInterrupt mask irq)

-- We don't need to acknowledge interrupts explicitly because we don't use
-- the vectored interrupt controller.
ackInterrupt :: Ptr CallbackData -> IRQ -> IO ()
ackInterrupt env irq = do
  callGICApi gic (GIC.ackInterrupt irq)
      where gic = GicState { env = env,
        gicDistBase = gicDistributorBase,
        gicIFBase = gicInterfaceBase }

foreign import ccall unsafe "qemu_run_devices"
    runDevicesCallback :: IO ()

getActiveIRQ :: Ptr CallbackData -> IO (Maybe IRQ)
getActiveIRQ env = do
    runDevicesCallback
    active <- callGICApi gicdata $ GIC.getActiveIRQ
    case active of
        Just 0x3FF -> return Nothing
        _ -> return active
      where gicdata = GicState { env = env,
        gicDistBase = gicDistributorBase,
        gicIFBase = gicInterfaceBase }

configureTimer :: Ptr CallbackData -> IO IRQ
configureTimer env = do
    MPT.callMPTimerApi mptdata $ MPT.mpTimerInit
    return timerIRQ
      where mptdata = MPT.MPTimerState { MPT.env = env,
        MPT.mptBase = mptBase }

initIRQController :: Ptr CallbackData -> IO ()
initIRQController env = callGICApi gicdata $ GIC.initIRQController
  where gicdata = GicState { env = env,
    gicDistBase = gicDistributorBase,
    gicIFBase = gicInterfaceBase }

resetTimer :: Ptr CallbackData -> IO ()
resetTimer env = do
    MPT.callMPTimerApi mptdata $ MPT.resetTimer
      where mptdata = MPT.MPTimerState { MPT.env = env,
        MPT.mptBase = mptBase }

isbCallback :: Ptr CallbackData -> IO ()
isbCallback _ = return ()

dsbCallback :: Ptr CallbackData -> IO ()
dsbCallback _ = return ()

dmbCallback :: Ptr CallbackData -> IO ()
dmbCallback _ = return ()

cacheCleanByVACallback :: Ptr CallbackData -> VPtr -> PAddr -> IO ()
cacheCleanByVACallback _cptr _mva _pa = return ()

cacheCleanByVA_PoUCallback :: Ptr CallbackData -> VPtr -> PAddr -> IO ()
cacheCleanByVA_PoUCallback _cptr _mva _pa = return ()

cacheInvalidateByVACallback :: Ptr CallbackData -> VPtr -> PAddr -> IO ()
cacheInvalidateByVACallback _cptr _mva _pa = return ()

cacheInvalidateByVA_ICallback :: Ptr CallbackData -> VPtr -> PAddr -> IO ()
cacheInvalidateByVA_ICallback _cptr _mva _pa = return ()

cacheInvalidate_I_PoUCallback :: Ptr CallbackData -> IO ()
cacheInvalidate_I_PoUCallback _ = return ()

cacheCleanInvalidateByVACallback ::
    Ptr CallbackData -> VPtr -> PAddr -> IO ()
cacheCleanInvalidateByVACallback _cptr _mva _pa = return ()

branchFlushCallback :: Ptr CallbackData -> VPtr -> PAddr -> IO ()
branchFlushCallback _cptr _mva _pa = return ()

cacheClean_D_PoUCallback :: Ptr CallbackData -> IO ()
cacheClean_D_PoUCallback _ = return ()

cacheCleanInvalidate_D_PoCCallback :: Ptr CallbackData -> IO ()
cacheCleanInvalidate_D_PoCCallback _ = return ()

cacheCleanInvalidate_D_PoUCallback :: Ptr CallbackData -> IO ()
cacheCleanInvalidate_D_PoUCallback _ = return ()

cacheCleanInvalidateL2RangeCallback ::
    Ptr CallbackData -> PAddr -> PAddr -> IO ()
cacheCleanInvalidateL2RangeCallback _ _ _ = return ()

cacheInvalidateL2RangeCallback :: Ptr CallbackData -> PAddr -> PAddr -> IO ()
cacheInvalidateL2RangeCallback _ _ _ = return ()

cacheCleanL2RangeCallback :: Ptr CallbackData -> PAddr -> PAddr -> IO ()
cacheCleanL2RangeCallback _ _ _ = return ()

cacheLine :: Int
cacheLine = error "see Kernel_Config.thy"

cacheLineBits :: Int
cacheLineBits = error "see Kernel_Config.thy"
