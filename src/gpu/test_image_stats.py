"""Reduced GPU metadata versus Letterbox/cut/finite reference decisions.

--host-only exercises buffer validation and cache logic without Vulkan. Device
checks permit only the small HOST_READ reduction buffers to reach the host.
"""
from __future__ import annotations
import ctypes
import pathlib
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src/ref"))
sys.path.insert(0,str(ROOT/"src/layer"))
import nr_daemon
import xmxres as X
import gpu_image_stats as G


class FakeBuffer:
    def __init__(self,runtime,count,dtype,kind):
        self.id=runtime.next_id;runtime.next_id+=1
        self._lib=runtime.lib
        self.address=self.id*4096
        self.nbytes=int(count)*np.dtype(dtype).itemsize
        self.kind=kind
        self.bytes=np.zeros(self.nbytes,np.uint8)
        runtime.buffers[self.address]=self

    def upload(self,values):
        raw=np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        self.bytes[:raw.size]=raw
        return self

    def download(self,dtype=np.float32,count=None):
        if self.kind!=X.HOST_READ:
            raise AssertionError("the metadata implementation tried to read image pixels")
        return self.bytes.view(dtype)[:count].copy()

    def free(self): self.id=-1


class FakeRuntime:
    def __init__(self):
        self.lib=object();self.next_id=1;self.buffers={};self.recorded=0
        self.calls=[];self.aborts=0

    def buffer(self,count,dtype=np.float32,kind=X.DEVICE):
        return FakeBuffer(self,count,dtype,kind)

    def begin(self): self.recorded=0
    def submit(self): pass
    def abort(self): self.aborts+=1


class FakePipeline:
    def __init__(self,runtime): self.rt=runtime

    def record(self,push,groups):
        p=G._Push.from_buffer_copy(bytes(push))
        self.rt.calls.append((p.mode,groups))
        source=self.rt.buffers[p.a].bytes
        target=self.rt.buffers[p.out].bytes
        if p.mode in (0,3):
            rgb=source.reshape(p.height,p.width,4)[...,:3]
            rows=rgb.max(axis=(1,2));columns=rgb.max(axis=(0,2))
            if p.mode==0:
                values=np.concatenate((rows,columns)).astype(np.uint32)
            else:
                top,bottom,left,right=p.bounds
                values=np.array((rows[top-1] if top else 0xffffffff,
                    rows[bottom] if bottom<p.height else 0xffffffff,
                    columns[left-1] if left else 0xffffffff,
                    columns[right] if right<p.width else 0xffffffff,
                    rows[top],rows[bottom-1],columns[left],columns[right-1]),np.uint32)
        else:
            a=source.view(np.float32)[:p.width*p.height*p.channels]
            if p.mode==1:
                b=self.rt.buffers[p.b].bytes.view(np.float32)[:a.size]
                items=np.abs(a-b)
            else: items=(~np.isfinite(a)).astype(np.float32)
            values=np.array([items[i:i+1024].sum(dtype=np.float32)
                             for i in range(0,a.size,1024)],np.float32).view(np.uint32)
        target[:values.nbytes]=values.view(np.uint8)


class HostChecks(unittest.TestCase):
    def setUp(self):
        self.rt=FakeRuntime()
        self.patch=patch.object(X,"ImagePipeline",lambda path:FakePipeline(self.rt))
        self.patch.start();self.addCleanup(self.patch.stop)
        self.stats=G.GPUImageStats(self.rt);self.addCleanup(self.stats.close)

    def test_push_abi_and_validation(self):
        self.assertEqual(ctypes.sizeof(G._Push),56)
        self.assertEqual(G._Push.bounds.offset,40)
        source=self.rt.buffer(10*12*4,np.uint8)
        for size in ((0,10),(1.0,10),(True,10),(0xffffffff,2)):
            with self.assertRaises(ValueError): self.stats.region(source,*size,37)
        with self.assertRaises(ValueError): self.stats.region(source,12,10,97)
        with self.assertRaises(ValueError): self.stats.region(source,12,11,37)
        source._lib=object()
        with self.assertRaises(ValueError): self.stats.region(source,12,10,37)
        source._lib=self.rt.lib
        source.free()
        with self.assertRaises(ValueError): self.stats.region(source,12,10,37)
        self.assertEqual(self.rt.calls,[])

    def test_cached_letterbox_matches_reference_through_changes(self):
        rng=np.random.default_rng(448)
        reference=nr_daemon.Letterbox()
        for height,width,fmt,top,bottom,left,right in (
                (41,73,37,4,37,0,73),(41,73,37,4,37,0,73),
                (41,73,37,0,41,0,73),(41,73,44,0,41,3,70),
                (42,74,43,2,39,0,74),(42,74,43,0,42,0,74),
                (42,74,50,3,38,5,69),(42,74,50,0,42,0,74)):
            raw=rng.integers(3,256,(height,width,4),dtype=np.uint8)
            raw[:top,:,:3]=2;raw[bottom:,:,:3]=0
            raw[:,:left,:3]=0;raw[:,right:,:3]=1
            colour=nr_daemon.decode(raw.tobytes(),width,height,fmt)
            expected=reference.region(colour,(width,height,fmt))
            source=self.rt.buffer(raw.size,np.uint8).upload(raw)
            try: self.assertEqual(self.stats.region(source,width,height,fmt),expected)
            finally: source.free()
        self.assertIn((3,1),self.rt.calls)
        self.assertEqual(self.rt.calls[0],(0,1))
        self.assertEqual(self.rt.calls[1],(3,1))

    def test_dark_alpha_and_asymmetric_lines(self):
        rng=np.random.default_rng(338)
        for height,width in ((1,1),(2,2),(21,37),(96,129)):
            for case in range(8):
                raw=rng.integers(0,256,(height,width,4),dtype=np.uint8)
                if case==0: raw[...,:3]=0
                if case==1: raw[...,:3]=2
                if case==2: raw[...,:3]=3
                if case>2:
                    margin=max(1,height//10)
                    raw[:margin,:,:3]=0
                    raw[-margin-(case%3):,:,:3]=0
                source=self.rt.buffer(raw.size,np.uint8).upload(raw)
                try:
                    expected=nr_daemon.active_region(nr_daemon.decode(raw.tobytes(),width,height,44))
                    # Match the cached CPU algorithm, including its own cache decision.
                    reference=nr_daemon.Letterbox()
                    reference.key=self.stats.key;reference.bounds=self.stats.bounds
                    expected=reference.region(nr_daemon.decode(raw.tobytes(),width,height,44),
                                              (width,height,44))
                    self.assertEqual(self.stats.region(source,width,height,44),expected)
                finally: source.free()

    def test_delta_and_nonfinite(self):
        rng=np.random.default_rng(239)
        a=rng.random((37,53,3),dtype=np.float32)
        b=rng.random(a.shape,dtype=np.float32)
        sa=self.rt.buffer(a.size).upload(a);sb=self.rt.buffer(b.size).upload(b)
        self.assertAlmostEqual(self.stats.delta(sa,sb,53,37),float(np.abs(a-b).mean()),places=7)
        self.assertEqual(self.stats.nonfinite(sa,53,37,3),0)
        a.reshape(-1)[[0,1023,1024,-1]]=[np.nan,np.inf,-np.inf,np.nan]
        sa.upload(a)
        self.assertEqual(self.stats.nonfinite(sa,53,37,3),4)
        with self.assertRaises(ValueError): self.stats.delta(sa,None,53,37)
        self.stats.close()
        with self.assertRaises(RuntimeError): self.stats.nonfinite(sa,53,37,3)

    def test_failed_recording_aborts(self):
        source=self.rt.buffer(8*8*4,np.uint8)
        with patch.object(self.stats.pipeline,"record",side_effect=RuntimeError("dispatch failed")):
            with self.assertRaisesRegex(RuntimeError,"dispatch failed"):
                self.stats.region(source,8,8,37)
        self.assertEqual(self.rt.aborts,1)


class DeviceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rt=X.Runtime()

    def setUp(self):
        self.stats=G.GPUImageStats(self.rt);self.addCleanup(self.stats.close)

    def test_device_regions_and_reductions_only_read_metadata(self):
        original=X.Buffer.download
        def reduced_only(buffer,*args,**kwargs):
            if buffer.kind!=X.HOST_READ:
                raise AssertionError("full image readback inside GPUImageStats")
            return original(buffer,*args,**kwargs)
        rng=np.random.default_rng(226)
        reference=nr_daemon.Letterbox()
        with patch.object(X.Buffer,"view",side_effect=AssertionError("host image view")),\
             patch.object(X.Buffer,"download",reduced_only):
            for height,width,fmt,top,bottom,left,right in (
                    (41,73,37,4,37,0,73),(41,73,37,4,37,0,73),
                    (41,73,37,0,41,0,73),(41,73,44,0,41,3,70),
                    (42,74,43,2,39,0,74),(42,74,43,0,42,0,74),
                    (42,74,50,3,38,5,69),(42,74,50,0,42,0,74),
                    (1,1,44,0,1,0,1),(192,320,37,0,192,0,320)):
                raw=rng.integers(3,256,(height,width,4),dtype=np.uint8)
                raw[:top,:,:3]=2;raw[bottom:,:,:3]=0
                raw[:,:left,:3]=0;raw[:,right:,:3]=1
                source=self.rt.buffer(raw.size,np.uint8,kind=X.DEVICE).upload(raw)
                try:
                    expected=reference.region(nr_daemon.decode(raw.tobytes(),width,height,fmt),
                                              (width,height,fmt))
                    self.assertEqual(self.stats.region(source,width,height,fmt),expected)
                finally: source.free()
            a=rng.normal(size=(33,61,4)).astype(np.float32)
            b=rng.normal(size=a.shape).astype(np.float32)
            sa=self.rt.buffer(a.size,kind=X.DEVICE).upload(a)
            sb=self.rt.buffer(b.size,kind=X.DEVICE).upload(b)
            try:
                self.assertAlmostEqual(self.stats.delta(sa,sb,61,33,4),float(np.abs(a-b).mean()),places=6)
                self.assertEqual(self.stats.nonfinite(sa,61,33,4),0)
                a.reshape(-1)[[0,1023,1024,-1]]=[np.nan,np.inf,-np.inf,np.nan]
                sa.upload(a)
                self.assertEqual(self.stats.nonfinite(sa,61,33,4),4)
            finally: sa.free();sb.free()


if __name__=="__main__":
    host_only="--host-only" in sys.argv
    sys.argv=[value for value in sys.argv if value!="--host-only"]
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(HostChecks)
    if not host_only: suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DeviceChecks))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
