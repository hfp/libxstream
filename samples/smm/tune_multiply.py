#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxstream/                     #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
import opentuner
from opentuner.search.manipulator import IntegerParameter
from opentuner.tuningrunmain import TuningRunMain
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.search import technique as ot_technique
from opentuner.search import bandittechniques as ot_bandit
from opentuner.search import differentialevolution as ot_de
from opentuner.search import evolutionarytechniques as ot_evo
from opentuner.search import simplextechniques as ot_simplex
from signal import signal, SIGINT
import tempfile
import hashlib
import copy
import json
import glob
import math
import sys
import re
import os
import ctypes
import ctypes.util
import random

default_enable_tune = {"tune", "enabled", "on"}
default_basename = "tune_multiply"
default_mnk = "23x23x23"
default_size = 30000  # batch size (S) the benchmark driver assumes
default_dbg = False
default_vlen = 8

# value of a tunable absent from a JSON-file (BS, BM, and BN are required)
default_params = {
    "BK": 0,
    "WS": 0,
    "WG": 0,
    "LU": 0,
    "NZ": 0,
    "AL": 0,
    "TB": 0,
    "TC": 1,
    "AP": 0,
    "AA": 0,
    "AB": 0,
    "AC": 0,
    "XF": 0,
}
# JSON entries that identify or measure a kernel rather than configure it
json_nonparams = {"DEVICE", "TYPEID", "M", "N", "K", "S", "GFLOPS"}

type_dp = 3  # TYPEID for double-precision
type_sp = 1  # TYPEID for single-precision
sp_dp_ratio = 0.5  # assumed SP:DP throughput ratio for normalization
# typename the benchmark reports for a TYPEID, i.e. the JSON-file's name
type_names = {type_dp: "double", type_sp: "float"}


def start(args):
    """Construct and start tuner instance"""
    instance = SmmTuner(args)
    if not default_dbg:
        try:
            TuningRunMain(instance, args).main()
        except Exception as e:
            print("IGNORED {}: {}".format(type(e).__name__, e))
            instance.save_final_config(None, True)
    else:
        TuningRunMain(instance, args).main()


def env_intvalue(env, default, lookup=True):
    value = (
        os.getenv(env, default)
        if lookup
        else env if env is not None else default
    )
    try:
        return int(value)
    except ValueError:
        return int(default)


def json_label(filename):
    """Kernel part of a JSON-file's name: type and shape"""
    match = re.match(
        r"\.?({}-[^-]+-\d+x\d+x\d+)".format(default_basename),
        os.path.basename(filename),
    )
    return match.group(1) if match else None


def json_name(label, data):
    """Name of a JSON-file: kernel, batch unless default, parameter hash"""
    size = data.get("S", default_size)
    batch = "-s{}".format(ilog2(size)) if size != default_size else ""
    params = {
        k: v
        for k, v in data.items()
        if k not in json_nonparams
        and (k not in default_params or default_params[k] != v)
    }
    text = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return "{}{}-{}.json".format(label, batch, digest)


def json_place(source, data):
    """Move a JSON-file to its name, joining a file of equal parameters"""
    result = source
    label = json_label(source)
    if label:
        target = os.path.normpath(
            os.path.join(os.path.dirname(source), json_name(label, data))
        )
        try:
            if target == os.path.normpath(source):
                result = target
            elif not os.path.exists(target):
                os.rename(source, target)  # keeps the file's age
                result = target
            else:  # same parameters: the higher GFLOPS stays
                with open(target, "r") as file:
                    other = json.load(file)
                if other.get("DEVICE") != data.get("DEVICE"):
                    print(
                        "WARNING: {} and {} are from different devices.".format(
                            source, target
                        )
                    )
                elif other.get("GFLOPS", 0) < data.get("GFLOPS", 0):
                    os.replace(source, target)
                    result = target
                else:
                    os.remove(source)
                    result = target
        except Exception:
            pass
    return result


def csv_expand(filename, jsondir, separator=";"):
    """Write the JSON-files a CSV-file was merged from (inverse of --merge)"""
    nwritten = nexists = nerror = 0
    with open(filename, "r") as csvfile:
        header = csvfile.readline().strip().split(separator)
        nhead = len(header)
        for line in csvfile:
            record = line.strip()
            if not record:
                continue
            fields = record.split(separator)
            if nhead != len(fields) and (nhead + 1) != len(fields):
                print("Failed to read {} of {}.".format(record, filename))
                nerror = nerror + 1
                continue
            entry = dict(zip(header, fields))
            # XF is vendor-specific and hence unnamed by the header
            if nhead < len(fields):
                entry["XF"] = fields[nhead]
            try:
                data = {"DEVICE": entry["DEVICE"]}  # identify the kernel
                for key in ("TYPEID", "M", "N", "K"):
                    data[key] = int(entry[key])
                # a merge writes zero for a measurement the JSON-file lacked,
                # and an absent S is what keeps the name free of a suffix
                gflops, size = float(entry["GFLOPS"]), int(entry["S"])
                if 0 < gflops:
                    data["GFLOPS"] = gflops
                if 0 < size:
                    data["S"] = size
                for key in entry:  # configure the kernel
                    if key not in json_nonparams:
                        data[key] = int(entry[key])
                typename = type_names.get(data["TYPEID"])
                if typename is None:
                    raise KeyError("TYPEID")
                label = "{}-{}-{}x{}x{}".format(
                    default_basename,
                    typename,
                    data["M"],
                    data["N"],
                    data["K"],
                )
                # a default XF is absent from a tuned JSON-file
                if 0 == data.get("XF", 0):
                    data.pop("XF", None)
            except (KeyError, ValueError):
                print("Failed to read {} of {}.".format(record, filename))
                nerror = nerror + 1
                continue
            target = os.path.join(jsondir, json_name(label, data))
            if os.path.exists(target):  # never reset a JSON-file's age
                nexists = nexists + 1
                continue
            with open(target, "w") as jsonfile:
                json.dump(data, jsonfile, sort_keys=True)
                jsonfile.write("\n")
            nwritten = nwritten + 1
    msg = "Wrote {} JSON-file(s) to {}".format(nwritten, jsondir)
    if 0 != nexists:
        msg = "{} ({} already existed)".format(msg, nexists)
    print(msg)
    return 1 if 0 != nerror else 0


def ilog2(n):
    i, t = (0 if 1 != n else 1), 1
    while t < n:
        t <<= 1
        i += 1
    return i


class Libxs(object):
    """Thin ctypes binding to the libxs prediction API (flat entry points)."""

    def __init__(self):
        self.lib = None
        name = "libxs.dylib" if "Darwin" == os.uname()[0] else "libxs.so"
        here = os.path.dirname(os.path.realpath(__file__))
        candidates = []
        env = os.getenv("LIBXS_LIB")
        if env:
            candidates.append(
                env if os.path.isfile(env) else os.path.join(env, name)
            )
        root = os.getenv("LIBXSROOT")
        if root:
            candidates.append(os.path.join(root, "lib", name))
        # the sibling checkout, as the Makefiles find it
        candidates.append(
            os.path.join(here, "..", "..", "..", "libxs", "lib", name)
        )
        found = ctypes.util.find_library("xs")
        if found:
            candidates.append(found)
        for path in candidates:
            if path and (os.path.isfile(path) or path == found):
                try:
                    self.lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    self.path = path
                    break
                except OSError:
                    self.lib = None
        if self.lib is not None:
            try:
                self._bind()
            except AttributeError:  # an older library without the flat API
                self.lib = None

    def _bind(self):
        vp, dp, ci = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        )
        lib = self.lib
        lib.libxs_init.argtypes, lib.libxs_init.restype = [], None
        lib.libxs_predict_create.argtypes = [ci, ci]
        lib.libxs_predict_create.restype = vp
        (
            lib.libxs_predict_destroy.argtypes,
            lib.libxs_predict_destroy.restype,
        ) = ([vp], None)
        lib.libxs_predict_push.argtypes = [vp, vp, dp, dp]
        lib.libxs_predict_push.restype = ci
        lib.libxs_predict_build.argtypes = [vp, ci, ci, ctypes.c_double]
        lib.libxs_predict_build.restype = ci
        lib.libxs_predict_eval_flat.argtypes = [vp, dp, dp, dp, dp, dp, dp]
        lib.libxs_predict_eval_flat.restype = None
        lib.libxs_init()

    def available(self):
        return self.lib is not None

    def build(self, rows, ninputs, noutputs):
        """A model built from (inputs, outputs) rows, or None."""
        lib, model = self.lib, None
        handle = lib.libxs_predict_create(ninputs, noutputs)
        if handle:
            ain, aout = ctypes.c_double * ninputs, ctypes.c_double * noutputs
            for xin, yout in rows:
                lib.libxs_predict_push(None, handle, ain(*xin), aout(*yout))
            # order 1: the outputs are measurements, not a smooth series
            if 0 == lib.libxs_predict_build(handle, 0, 1, 0.0):
                model = handle
            else:
                lib.libxs_predict_destroy(handle)
        return model

    def evaluate(self, model, xin, noutputs):
        """(values, confidence, variance) for one input row."""
        arr = ctypes.c_double * noutputs
        out, conf, var = arr(), arr(), arr()
        self.lib.libxs_predict_eval_flat(
            model,
            (ctypes.c_double * len(xin))(*xin),
            out,
            conf,
            var,
            None,
            None,
        )
        return list(out), list(conf), list(var)

    def destroy(self, model):
        if model:
            self.lib.libxs_predict_destroy(model)


class LibxsSurrogate(ot_technique.SearchTechnique):
    """
    Model-based proposals for one tuning run. Every trial, whoever requested it,
    trains a surrogate of two outputs: the throughput reached and whether the
    configuration built and validated at all (failures are common, and a
    proposer that cannot learn them keeps proposing them). A proposal is the
    candidate with the best predicted throughput times probability of being
    valid, plus an exploration bonus of LIBXS_SURROGATE_KAPPA standard
    deviations. The bandit around it decides whether that earns trials.

    The standard deviation is the spread among a candidate's nearest measured
    neighbours, so it is local: a region nobody has sampled does not look
    uncertain, it looks like its slow neighbours, and such a surrogate exploits
    well and discovers at chance. LIBXS_SURROGATE_NOVELTY therefore adds a
    bonus for distance from everything measured, in units of the best
    throughput so far; zero leaves the exploitation-only proposer.
    """

    libxs = None

    def __init__(self, *pargs, **kwargs):
        super(LibxsSurrogate, self).__init__(*pargs, **kwargs)
        self.kappa = float(os.getenv("LIBXS_SURROGATE_KAPPA", "1.0"))
        self.ncand = int(os.getenv("LIBXS_SURROGATE_CANDIDATES", "512"))
        self.warmup = int(os.getenv("LIBXS_SURROGATE_WARMUP", "8"))
        self.novelty = float(os.getenv("LIBXS_SURROGATE_NOVELTY", "1.0"))
        self.params, self.rows, self.seen = None, [], set()
        self.model, self.dirty, self.best = None, False, None

    def tunables(self, cfg):
        if self.params is None:  # the fixed parameters carry no signal
            self.params = [
                p
                for p in self.manipulator.parameters(cfg)
                if hasattr(p, "min_value") and p.min_value != p.max_value
            ]
        return self.params

    def key(self, cfg):
        return tuple(p.get_value(cfg) for p in self.tunables(cfg))

    def on_result(self, result):
        cfg = result.configuration.data
        xin = [float(v) for v in self.key(cfg)]
        gflops = float(result.accuracy or 0.0)
        valid = 1.0 if 0 < gflops else 0.0
        self.rows.append((xin, [gflops, valid]))
        self.seen.add(tuple(xin))
        if self.best is None or self.best[0] < gflops:
            self.best = (gflops, dict(cfg))
        self.dirty = True

    def mutate(self, cfg):
        out = dict(cfg)
        for p in random.sample(self.params, max(1, len(self.params) // 3)):
            lo, hi = p.min_value, p.max_value
            step = max(1, (hi - lo) // 8)
            v = p.get_value(out) + random.randint(-step, step)
            p.set_value(out, min(hi, max(lo, v)))
        return out

    def desired_configuration(self):
        cfg = self.manipulator.random()
        if len(self.rows) >= self.warmup and self.libxs.available():
            if self.dirty:
                self.libxs.destroy(self.model)
                self.model = self.libxs.build(
                    self.rows, len(self.tunables(cfg)), 2
                )
                self.dirty = False
            if self.model:
                # each parameter scaled to its range, so none dominates distance
                span = [
                    (p.min_value, max(1, p.max_value - p.min_value))
                    for p in self.params
                ]

                def unit(x):
                    return [(v - lo) / w for v, (lo, w) in zip(x, span)]

                seen = (
                    [unit(x) for x, _ in self.rows] if 0 < self.novelty else []
                )
                scale = max(1.0, self.best[0]) if self.best else 1.0
                pool = [
                    self.manipulator.random() for _ in range(self.ncand // 2)
                ]
                if self.best is not None:
                    pool += [
                        self.mutate(self.best[1])
                        for _ in range(self.ncand - len(pool))
                    ]
                bestscore, choice = None, None
                for cand in pool:
                    xin = [float(v) for v in self.key(cand)]
                    if tuple(xin) in self.seen:
                        continue
                    val, conf, var = self.libxs.evaluate(self.model, xin, 2)
                    # a 0/1 output answers a class with a share behind it
                    pvalid = conf[1] if 0.5 <= val[1] else 1.0 - conf[1]
                    score = max(0.0, val[0]) * pvalid + self.kappa * math.sqrt(
                        max(0.0, var[0])
                    )
                    if seen:
                        u = unit(xin)
                        near = min(
                            sum((a - b) * (a - b) for a, b in zip(u, v))
                            for v in seen
                        )
                        score += self.novelty * scale * math.sqrt(near)
                    if bestscore is None or bestscore < score:
                        bestscore, choice = score, cand
                if choice is not None:
                    cfg = choice
        return cfg


LibxsSurrogate.libxs = Libxs()
if LibxsSurrogate.libxs.available():
    # the default bandit's members, and the same members plus the surrogate: the
    # pair is the comparison, so nothing else about the ensemble may differ
    ot_technique.register(
        ot_bandit.AUCBanditMetaTechnique(
            [
                ot_de.DifferentialEvolutionAlt(),
                ot_evo.UniformGreedyMutation(),
                ot_evo.NormalGreedyMutation(mutation_rate=0.3),
                ot_simplex.RandomNelderMead(),
                LibxsSurrogate(),
            ],
            name="LibxsBanditA",
        )
    )
else:
    sys.stderr.write(
        "WARNING: libxs not loadable (set LIBXS_LIB), LibxsBanditA is absent.\n"
    )


class SmmTuner(MeasurementInterface):
    def __init__(self, args):
        """Setup common state and define search space"""
        super(SmmTuner, self).__init__(args)
        mnk = tuple(max(int(i), 1) for i in self.args.mnk.split("x"))
        self.mnk = (mnk + (mnk[0], mnk[0]))[:3]
        self.wsx = self.mnk[0] * self.mnk[1]
        self.manip = ConfigurationManipulator()
        # sanitize input arguments
        self.args.mb = max(self.args.mb, 1)
        self.args.bs = max(min(self.args.bs, self.args.mb), 1)
        self.args.bm = [max(self.args.bm, 1), self.mnk[0]][0 == self.args.bm]
        self.args.bn = [max(self.args.bn, 1), 1][0 == self.args.bn]
        self.args.bk = [max(self.args.bk, 1), self.mnk[2]][0 == self.args.bk]
        self.args.ws = min(self.args.ws, self.wsx)
        self.gfbase = self.gfsave = self.gflops = self.gflogs = self.gfscnt = 0
        self.config = self.typename = self.typeid = self.device = self.size = (
            None
        )
        self.bs = self.bm = self.bn = self.bk = self.ws = self.wg = self.lu = (
            None
        )
        self.nz = self.al = self.tb = self.tc = None
        self.ap = self.aa = self.ab = self.ac = self.xf = None
        self.idevice, self.ndevices = None, 0
        self.exepath = os.path.join(
            os.path.dirname(os.path.realpath(sys.argv[0])), "acc_bench.x"
        )
        runcmd = self.launch(["LIBXSTREAM_VERBOSE=2"], 0, nrep=1)
        self.run_result = (  # verbosity to capture device name and tuned parameters
            self.call_program(" ".join(runcmd))
            if (  # consider validating parameters during merge
                (self.args.merge is None or 0 > self.args.merge)
                or (self.args.check is None or 0 != self.args.check)
            )
            and (self.args.update is None or "" == self.args.update)
            else None
        )
        if self.run_result:
            stdout = str(self.run_result["stdout"])
            if 0 >= self.args.size:
                sizepat = "(\\w+)\\s+[0-9]+\\s+([0-9]+)"
                size = re.search(sizepat, stdout)
                self.size = int(size.group(2)) if size and size.group(2) else 0
            else:
                self.size = self.args.size
            typename = re.search(
                "typename \\(id=([0-9]+)\\):\\s+(\\w+)", stdout
            )
            self.typename = (
                typename.group(2) if typename and typename.group(2) else ""
            )
            self.typeid = (
                int(typename.group(1)) if typename and typename.group(1) else 0
            )
            devicepat = 'INFO ACC/OpenCL:\\s+ndevices=([0-9]+)\\s+device[0-9]+="([^"]+)"'
            device = re.search(devicepat, str(self.run_result["stderr"]))
            self.ndevices = (
                int(device.group(1)) if device and device.group(1) else 0
            )
            self.device = device.group(2) if device and device.group(2) else ""
            # idevice: make certain resources/names unique on a per-rank basis
            envrank_mpich = os.getenv("PMI_RANK")  # global
            envrank_ompi = os.getenv(
                "OMPI_COMM_WORLD_LOCAL_RANK", envrank_mpich
            )
            envrank = os.getenv("MPI_LOCALRANKID", envrank_ompi)
            if envrank:
                self.idevice = int(envrank) % self.ndevices
        elif self.args.update is not None and "" != self.args.update:
            self.device = self.args.update
        if self.run_result and 0 == self.run_result["returncode"]:
            seedpat = "INFO ACC/LIBSMM:\\s+SMM-kernel\\s+{}={}\\s+gen=".format(
                "{t,m,n,k, bs,bm,bn,bk, ws,wg, lu,nz,al, tb,tc, ap,aa,ab,ac}",
                "{{{}, {}}}".format(  # key and value
                    "{},{}".format(  # t,m,n,k (key)
                        self.typeid, ",".join(map(str, self.mnk))
                    ),
                    "{}, {}, {}, {}, {}".format(  # value: if neg. "-*[0-9]+"
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)",  # bs,bm,bn,bk
                        "(-*[0-9]+),(-*[0-9]+)",  # ws,wg
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)",  # lu,nz,al
                        "(-*[0-9]+),(-*[0-9]+)",  # tb,tc
                        "(-*[0-9]+),(-*[0-9]+),(-*[0-9]+),(-*[0-9]+)(, .+)*",  # ap,aa,ab,ac[, ext]
                    ),
                ),
            )
            seed = re.search(seedpat, str(self.run_result["stderr"]))
            nprm = len(seed.groups()) if seed else 0
            if 15 > nprm:
                print("WARNING: missed to parse initial parameters!")
            maxlu = (self.mnk[0] + default_vlen - 1) // default_vlen
            # setup fixed and tunable parameters
            params, paramt = [], []
            self.create_param("BS", params, paramt, seed, 1, 1, self.args.mb)
            self.create_param("BM", params, paramt, seed, 2, 1, self.mnk[0])
            self.create_param("BN", params, paramt, seed, 3, 1, self.mnk[1])
            self.create_param("BK", params, paramt, seed, 4, 1, self.mnk[0])
            self.create_param("WS", params, paramt, seed, 5, 1, self.wsx)
            self.create_param(
                "WG", params, paramt, seed, 6, -2, 1, False
            )  # avoid WG=2
            self.create_param("LU", params, paramt, seed, 7, -2, maxlu)
            self.create_param("NZ", params, paramt, seed, 8, 0, 1)
            self.create_param("AL", params, paramt, seed, 9, 0, 1)
            self.create_param("TB", params, paramt, seed, 10, 0, 1)
            self.create_param("TC", params, paramt, seed, 11, 0, 1)
            self.create_param("AP", params, paramt, seed, 12, 0, 1)
            self.create_param("AA", params, paramt, seed, 13, 0, 2)
            self.create_param("AB", params, paramt, seed, 14, 0, 2)
            self.create_param("AC", params, paramt, seed, 15, 0, 1)
            if 15 < nprm and seed.group(16) and 2 < len(seed.group(16)):
                self.create_param(
                    "XF", params, paramt, seed.group(16)[2:], -1, 0, 1
                )
            else:
                self.create_param("XF", params, paramt, 0, -1, 0, 1)
            if not paramt:
                sys.tracebacklimit = 0
                raise RuntimeError(
                    "All parameters are fixed with environment variables!"
                )
            for param in params + paramt:
                self.manip.add_parameter(param)
        if (
            (  # consider to update and/or merge JSONS (update first)
                self.args.merge is not None
                and (0 <= self.args.merge or self.typeid)
                and (
                    (self.args.check is not None and 0 == self.args.check)
                    or (self.run_result and 0 == self.run_result["returncode"])
                )
            )
            or (self.args.check is None or 0 != self.args.check)
            or (self.args.update is None or "" != self.args.update)
        ):
            filepattern = "{}-*.json".format(default_basename)
            filedot = "." + filepattern
            filenames = glob.glob(
                os.path.normpath(os.path.join(self.args.jsondir, filedot))
            )
            for filename in filenames:
                self.rename_dotfile(filename)
            filenames = glob.glob(
                os.path.normpath(os.path.join(self.args.jsondir, filepattern))
            )
            do_update = self.args.update is None or "" != self.args.update
            do_check = self.args.check is None or 0 != self.args.check
            do_merge = self.args.merge is not None
            if do_update:
                self.update_jsons(filenames)
            elif do_check:
                self.update_jsons(filenames)
                if 0 < self.gfscnt and self.args.check and 0 > self.args.check:
                    gmn = math.exp(self.gflogs / self.gfscnt)
                    print("Geometric mean of {} GFLOPS/s".format(round(gmn)))
            elif do_merge:
                self.merge_jsons(self.migrate_jsons(filenames))
            exit(0)
        elif (
            (self.typename and "" != self.typename)
            and (self.device and "" != self.device)
            and (self.typeid and 0 < self.ndevices)
            and (self.size and 0 < self.size)
        ):  # setup database (DB)
            if self.args.database is None:  # adjust DB-location
                tmpdir = os.path.join(tempfile.gettempdir(), "opentuner")
                if self.idevice is not None:
                    tmpdir += str(self.idevice)
                try:
                    os.mkdir(tmpdir)
                except Exception:
                    pass
                self.args.database = "sqlite:///" + os.path.join(
                    tmpdir, "{}.db".format(os.getpid())
                )
            if not self.args.label:  # label for DB-session
                self.args.label = "{}-{}-{}-s{}".format(
                    default_basename,
                    self.typename,
                    "x".join(map(str, self.mnk)),
                    ilog2(self.size),
                )
        else:
            sys.tracebacklimit = 0
            raise RuntimeError("Setup failed for {}!".format(self.exepath))
        # register signal handler (CTRL-C)
        signal(SIGINT, self.handle_sigint)
        self.handle_sigint_counter = 0

    def manipulator(self):
        return self.manip

    def create_param(
        self,
        name,
        params,
        paramt,
        match,
        match_id,
        value0,
        value1,
        expand=True,
    ):
        """Append integer-parameter to either params or paramt list"""
        value_key = "OPENCL_LIBSMM_SMM_{}".format(name)
        value_env = os.getenv(value_key)
        attribute = getattr(self, name.lower(), None)
        tunable = value_env in default_enable_tune
        if 0 <= match_id:
            if match and match.group(match_id):
                value = int(match.group(match_id))
            else:
                value = 0 if value_env is None else int(value_env)
            if not tunable:
                tunable = value_env is None
        else:
            if attribute is None:
                value = getattr(self.args, name.lower(), None)
                if value is None:
                    value = int(value_env if match is None else match)
            else:
                value = int(attribute)
            if not tunable:
                tunable = value_env is None and 0 != value
        if tunable:  # consider expanding value range according to seed
            v0 = min(value0, value) if expand else value0
            v1 = max(value1, value) if expand else value1
            paramt.append(IntegerParameter(name, v0, v1))
        else:  # fixed parameter
            params.append(IntegerParameter(name, value, value))
        if attribute is None:
            setattr(self, name.lower(), value)

    def launch(self, envs, check=None, nrep=None, verbose=None):
        """Launch executable supplying environment and arguments"""
        envlist = envs if isinstance(envs, list) else self.environment(envs)
        mnk = (envs["M"], envs["N"], envs["K"]) if "M" in envs else self.mnk
        env_exe = " ".join(map(str, envlist))
        if verbose is not None and 0 != int(verbose):
            msg = env_exe.replace("OPENCL_LIBSMM_SMM_", "")
            print("{}: {}".format("x".join(map(str, mnk)), msg))
        env_std = "OMP_PROC_BIND=TRUE OPENCL_LIBSMM_SMM_S=0"
        env_jit = "NEO_CACHE_PERSISTENT=0 CUDA_CACHE_DISABLE=1"
        env_check = "CHECK={}".format(check if check is not None else 1)
        env_intrn = "{} {}".format(  # consider device-id
            (
                ""
                if self.idevice is None
                else "LIBXSTREAM_DEVICE={}".format(self.idevice)
            ),
            "{} {} {}".format(env_std, env_jit, env_check),  # environment
        ).strip()
        arg_exe = "{} {} {}".format(
            self.args.r if nrep is None else nrep,
            self.size if self.size else self.args.size,
            " ".join(map(str, mnk)),
        ).strip()
        return [env_exe, env_intrn, self.exepath, arg_exe]

    def seed_configurations(self):
        return [
            {
                "BS": self.bs if self.bs is not None else self.args.bs,
                "BM": self.bm if self.bm is not None else self.args.bm,
                "BN": self.bn if self.bn is not None else self.args.bn,
                "BK": self.bk if self.bk is not None else self.args.bk,
                "WS": self.ws if self.ws is not None else self.args.ws,
                "WG": self.wg if self.wg is not None else self.args.wg,
                "NZ": self.nz if self.nz is not None else self.args.nz,
                "LU": self.lu if self.lu is not None else self.args.lu,
                "AL": self.al if self.al is not None else self.args.al,
                "TB": self.tb if self.tb is not None else self.args.tb,
                "TC": self.tc if self.tc is not None else self.args.tc,
                "AP": self.ap if self.ap is not None else self.args.ap,
                "AA": self.aa if self.aa is not None else self.args.aa,
                "AB": self.ab if self.ab is not None else self.args.ab,
                "AC": self.ac if self.ac is not None else self.args.ac,
                "XF": self.xf if self.xf is not None else 0,
            }
        ]

    def objective(self):
        if 0 == self.args.tlevel:
            return opentuner.search.objective.MaximizeAccuracyMinimizeSize()
        else:
            return opentuner.search.objective.MaximizeAccuracy()

    def environment(self, config):
        return [
            "OPENCL_LIBSMM_SMM_{}={}".format(key, config[key])
            for key in sorted(config.keys())
            if 2 == len(key)
        ]

    def run(
        self, desired_result, input=None, limit=None, message=None, nrep=None
    ):  # noqa: A002
        """Run a configuration and return performance"""
        try:
            config = desired_result.configuration.data
            mnk = self.mnk
        except AttributeError:
            config = desired_result
            mnk = (config["M"], config["N"], config["K"])
        skip = False
        if self.args.quick:
            if 1 == config["AA"] or 1 == config["AB"]:
                skip = True
        performance = None
        if not skip:
            runcmd = self.launch(
                config, self.args.check, nrep, self.args.verbose
            )
            tlimit = self.args.timeout if 0 < self.args.timeout else None
            self.run_result = self.call_program(" ".join(runcmd), limit=tlimit)
            result = self.run_result["returncode"] if self.run_result else 1
            if 0 == result:
                performance = re.search(
                    "device:\\s+([0-9]+[^ ]*) ms\\s+([0-9]+[^ ]*)",
                    str(self.run_result["stdout"]),
                )
        if performance and performance.group(1) and performance.group(2):
            mseconds, gflops = float(performance.group(1)), float(
                performance.group(2)
            )
            if 0 < gflops:
                self.gflogs = self.gflogs + math.log(gflops)
                self.gfscnt = self.gfscnt + 1
            if config is not desired_result:
                kernelreq = round(
                    (100.0 * config["BM"] * config["BN"]) / self.wsx
                )
                # gflops are reported as "accuracy" (console output)
                result = Result(time=mseconds, accuracy=gflops, size=kernelreq)
                if (
                    self.gflops < gflops
                ):  # keep best config in case of early exit
                    self.config = desired_result.configuration
                    self.gflops = gflops
                    if 0 != self.gfbase:
                        self.save_final_config(self.config, final=False)
                    else:  # seed configuration
                        self.gfbase = gflops
            elif not self.args.verbose:
                if message:
                    status = "OK"
                    if 1 != int(nrep):
                        gfbase = config["GFLOPS"] if "GFLOPS" in config else 0
                        if 0 < gfbase:
                            status = "{}x - ".format(round(gflops / gfbase, 2))
                        status = status + "{} GFLOPS/s".format(round(gflops))
                    print("{} - {}".format(message, status), flush=True)
                else:
                    print(".", end="", flush=True)
        elif not skip:  # return non-competitive/bad result in case of an error
            failed = runcmd[0].replace("OPENCL_LIBSMM_SMM_", "")
            if message:
                msg = "{} - FAILED".format(message)
            else:
                msg = "FAILED[{}] {}: {}".format(
                    result, "x".join(map(str, mnk)), failed
                )
            if config is not desired_result:
                result = Result(time=float("inf"), accuracy=0.0, size=100.0)
            elif not self.args.verbose and not message:
                print("")
            print(msg, flush=True)
        else:
            result = Result(time=float("inf"), accuracy=0.0, size=100.0)
        return result

    def run_check(self, data):
        """Run configuration and return process return code (0 on success)"""
        self.run(data, nrep=1)
        result = self.run_result
        return result["returncode"] if result else 1

    def update_jsons(self, filenames):
        """Update device name or verify all JSONs"""
        if self.device:
            n = len(filenames)
            for i, filename in enumerate(filenames):
                try:
                    with open(filename, "r") as file:
                        data = json.load(file)
                    if self.args.check is None or 0 != self.args.check:
                        progress, r = (
                            "[{}/{}]: {}".format(i + 1, n, filename),
                            1,
                        )
                        if self.args.check is not None:
                            r = max(self.args.check, 0)
                        if "TYPEID" in data and self.typeid == data["TYPEID"]:
                            self.run(data, message=progress, nrep=r)
                    elif "DEVICE" in data and data["DEVICE"] != self.device:
                        print(
                            "Updated {} to {}.".format(filename, self.device)
                        )
                        data.update({"DEVICE": self.device})
                        with open(filename, "w") as file:
                            json.dump(data, file, sort_keys=True)
                            file.write("\n")
                except (json.JSONDecodeError, KeyError):
                    print("Failed to update {}.".format(filename))
        else:
            print("Cannot determine device name.")

    def make_csv_record(self, data, filename):
        """Make key-value tuples from JSON-data"""
        device = data["DEVICE"] if "DEVICE" in data else self.device
        value = (
            data["S"] if "S" in data else 0,  # pseudo key component
            data["GFLOPS"] if "GFLOPS" in data else 0,
            data["BS"],
            data["BM"],
            data["BN"],
        ) + tuple(data.get(k, v) for k, v in default_params.items())
        value = value + (filename,)  # last entry
        return (device, data["TYPEID"], data["M"], data["N"], data["K"]), value

    def merge_collect(self, filenames):
        """Collect merge candidates from JSON files"""
        merged, retain, delete = dict(), dict(), []
        skipcnt = 0
        for filename in filenames:
            try:
                with open(filename, "r") as file:
                    data = json.load(file)
                if not data or (
                    self.args.merge is not None
                    and (
                        (0 > self.args.merge and self.typeid != data["TYPEID"])
                        or (1 == self.args.merge and type_sp != data["TYPEID"])
                        or (2 == self.args.merge and type_dp != data["TYPEID"])
                    )
                ):
                    skipcnt = skipcnt + 1
                    continue
                key, value = self.make_csv_record(data, filename)
            except (json.JSONDecodeError, KeyError, TypeError):
                print("Failed to merge {} into CSV-file.".format(filename))
                continue
            except Exception:
                continue
            if bool(data) and key in merged:
                gfbase, mname = merged[key][1], merged[key][-1]
                gflops, mtime = value[1], os.path.getmtime(mname)
                if gfbase < gflops:  # merged data is worse
                    if mtime < os.path.getmtime(filename):  # older
                        delete.append(mname)
                    else:
                        if key in retain:
                            if retain[key][1] < gfbase:
                                delete.append(retain[key][-1])
                                retain[key] = merged[key]
                            else:
                                delete.append(mname)
                        else:
                            retain[key] = merged[key]
                else:  # merged data is leading
                    if mtime < os.path.getmtime(filename):  # older
                        if key in retain:
                            if retain[key][1] < gflops:
                                delete.append(retain[key][-1])
                                retain[key] = value
                            else:
                                delete.append(filename)
                        else:
                            retain[key] = value
                    else:  # newer
                        delete.append(filename)
                    data = dict()  # ensure data is not merged
            if bool(data) and (
                (self.args.check is not None and 0 == self.args.check)
                or 0 == self.run_check(data)
            ):
                merged[key] = value
        return merged, retain, delete, skipcnt

    def merge_write_csv(self, merged, total, skipcnt):
        """Write merged data to CSV and print summary"""
        if bool(merged):
            with open(self.args.csvfile, "w") as csvfile:
                csvfile.write(
                    "{}{}{}{}{}{}{}{}{}\n".format(
                        self.args.csvsep.join(
                            ["DEVICE", "TYPEID", "M", "N", "K"]
                        ),
                        self.args.csvsep,
                        "S",
                        self.args.csvsep,
                        self.args.csvsep.join(
                            ["GFLOPS", "BS", "BM", "BN", "BK"]
                        ),
                        self.args.csvsep,
                        self.args.csvsep.join(["WS", "WG", "LU", "NZ", "AL"]),
                        self.args.csvsep,
                        self.args.csvsep.join(
                            ["TB", "TC", "AP", "AA", "AB", "AC"]
                        ),
                    )
                )
                types = [key[1] for key in merged.keys()]
                pure = min(types) == max(types)
                for key, value in sorted(merged.items()):
                    gflops = (
                        value[1]
                        if pure or type_sp != key[1]
                        else value[1] * sp_dp_ratio
                    )
                    values = list(value[:-1])
                    if self.args.nogflops:
                        values[1] = 0
                    if 0 < gflops:
                        self.gflogs = self.gflogs + math.log(gflops)
                        self.gfscnt = self.gfscnt + 1
                    strkey = self.args.csvsep.join([str(k) for k in key])
                    strval = self.args.csvsep.join([str(v) for v in values])
                    csvfile.write(
                        "{}{}{}\n".format(strkey, self.args.csvsep, strval)
                    )
        msg = "Merged {} of {} JSONs into {}".format(
            len(merged), total - skipcnt, self.args.csvfile
        )
        if 0 < self.gfscnt:
            gmn = math.exp(self.gflogs / self.gfscnt)
            msg = "{} (geometric mean of {} GFLOPS/s)".format(msg, round(gmn))
        if not self.args.verbose and (
            self.args.check is None or 0 != self.args.check
        ):
            print("")
        print(msg)

    def migrate_jsons(self, filenames):
        """Rename JSON-files to their name (parameter hash), keeping age"""
        result, renamed = [], 0
        for filename in filenames:
            placed = filename
            try:
                with open(filename, "r") as file:
                    placed = json_place(filename, json.load(file))
            except Exception:
                pass
            renamed = renamed + (1 if placed != filename else 0)
            if placed not in result and os.path.exists(placed):
                result.append(placed)
        if 0 < renamed:
            print("Renamed {} JSON-files.".format(renamed))
        return result

    def merge_jsons(self, filenames):
        """Merge all JSONs into a single CSV-file"""
        if not self.args.csvfile or (
            self.idevice is not None and 0 != self.idevice
        ):
            return
        self.gflogs = self.gfscnt = 0
        merged, retain, delete, skipcnt = self.merge_collect(filenames)
        if "new" == self.args.prefer:
            losslog, losscnt = 0, 0
            for key, value in retain.items():
                if key in merged:
                    rname, mname = value[-1], merged[key][-1]
                    if os.path.getmtime(mname) < os.path.getmtime(rname):
                        retain[key] = merged[key]
                        merged[key] = value
                        gf_old = retain[key][1]
                        gf_new = merged[key][1]
                        if 0 < gf_old and 0 < gf_new:
                            ratio = gf_new / gf_old
                            losslog = losslog + math.log(ratio)
                            losscnt = losscnt + 1
                            if self.args.verbose:
                                mnk = "x".join(map(str, key[2:]))
                                print(
                                    "{}: {} -> {} GFLOPS/s ({:.1f}%)".format(
                                        mnk,
                                        round(gf_old),
                                        round(gf_new),
                                        100.0 * (ratio - 1),
                                    )
                                )
            if 0 < losscnt:
                gmn = math.exp(losslog / losscnt)
                print(
                    "Prefer new: {:.2f}x (geometric mean over {} kernels)".format(
                        gmn, losscnt
                    )
                )
        delete = delete + [v[-1] for v in retain.values()]
        if bool(delete):
            num, lst = len(delete), " ".join(delete)
            if self.args.delete:
                for filename in delete:
                    try:
                        os.remove(filename)
                    except Exception:
                        pass
                msg = "Removed {}".format(num)
                skipcnt = skipcnt + num
            else:  # the merge reports what -d would have removed
                msg = "Remove {} (use -d)".format(num)
            print("{}: {}".format(msg, lst))
            print("")
        self.merge_write_csv(merged, len(filenames), skipcnt)

    def rename_dotfile(self, dotfile):
        try:
            data = None
            with open(dotfile, "r") as file:
                data = json.load(file)
            gflops = data["GFLOPS"] if data and "GFLOPS" in data else 0
            if 0 < gflops:
                json_place(dotfile, data)
        except Exception:
            pass

    def save_final_config(self, configuration, final=True):
        """Called at termination"""
        if not final and (0 >= self.gflops or not configuration):
            return  # nothing to save
        config = configuration.data if configuration else None
        cfgenv = self.environment(config) if config else None
        envchk = os.getenv("CHECK")  # force CHECKing result unless CHECK=0
        result = (
            self.run_result["returncode"] if config and self.run_result else 1
        )
        if (
            0 == result
            and 0 == self.args.check
            and (envchk is None or "0" != envchk)
        ):
            tlimit = self.args.timeout if 0 < self.args.timeout else None
            self.run_result = self.call_program(
                " ".join(self.launch(cfgenv, 1)), limit=tlimit
            )
            result = self.run_result["returncode"] if self.run_result else 1
        # extend result for easier reuse
        if config:
            config["DEVICE"] = self.device
            config["GFLOPS"] = self.gflops
            config["TYPEID"] = self.typeid
            config["M"] = self.mnk[0]
            config["N"] = self.mnk[1]
            config["K"] = self.mnk[2]
            config["S"] = self.size
        filedev = "" if self.idevice is None else "-{}".format(self.idevice)
        filedot = os.path.join(
            self.args.jsondir, ".{}{}.json".format(self.args.label, filedev)
        )
        if config and self.gfsave < self.gflops:  # save intermediate result
            if 0 == self.gfsave and os.path.exists(filedot):  # backup
                self.rename_dotfile(filedot)
            # self.manipulator().save_to_file(config, filename)
            with open(filedot, "w") as file:
                cfg = config
                if "XF" in config and 0 == config["XF"]:
                    cfg = copy.deepcopy(config)
                    del cfg["XF"]
                json.dump(cfg, file, sort_keys=True)
                file.write("\n")  # append newline at EOF
            self.gfsave = self.gflops
        # check return code (consider not saving parameters)
        if 0 != result and not final:  # incorrect result
            failed = " ".join(map(str, cfgenv)).replace(
                "OPENCL_LIBSMM_SMM_", ""
            )
            mnk = "x".join(map(str, self.mnk))
            print("FAILED[{}] {}: {}".format(result, mnk, failed), flush=True)
            return
        if final and 0 < self.gflops and os.path.exists(filedot):
            filepattern = "{}-*.json".format(default_basename)
            filenames = glob.glob(
                os.path.normpath(os.path.join(self.args.jsondir, filepattern))
            )
            if not filenames and glob.glob(self.args.csvfile):
                msg = "WARNING: no JSON-file found but {} will be overwritten."
                print(msg.format(self.args.csvfile))
            filename = filedot
            try:
                with open(filedot, "r") as file:
                    filename = json_place(filedot, json.load(file))
            except Exception:
                pass
            if filename != filedot and filename not in filenames:
                filenames.append(filename)
                self.merge_jsons(filenames)
            speedup = round(
                (self.gflops / self.gfbase) if 0 < self.gfbase else 0, 1
            )
            msg = " ({}x over seed)".format(speedup) if 1 < speedup else ""
            print("Result{} was written to {}".format(msg, filename))
        elif final and self.args.merge is None:
            print("WARNING: no tuned results produced!")

    def handle_sigint(self, signum, frame):
        """Handle SIGINT or CTRL-C"""
        if 1 > self.handle_sigint_counter:  # avoid recursion
            self.handle_sigint_counter = self.handle_sigint_counter + 1
            msg = "\nWARNING: tuning {}-kernel interrupted."
            print(msg.format("x".join(map(str, self.mnk))))
            try:
                self.save_final_config(self.config, True)
            except Exception:
                pass
        exit(1)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    # adjust default value of existing arguments
    argparser.set_defaults(no_dups=True)
    # add primary arguments (parsed first)
    argparser.add_argument(
        "mnk",
        type=str,
        default=None,
        nargs="?",
        help="Shape (MxNxK), file of shapes, or JSON-directory (merge)",
    )
    argparser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=0,
        nargs="?",
        dest="r",
        help="Repetitions per experiment",
    )
    argparser.add_argument(
        "-e",
        "--csv-separator",
        type=(lambda c: c if isinstance(c, str) and 1 == len(c) else False),
        default=";",
        nargs="?",
        dest="csvsep",
        help="Separator used in CSV-file",
    )
    argparser.add_argument(
        "-o",
        "--csv-filename",
        type=str,
        default="{}.csv".format(default_basename),
        nargs="?",
        dest="csvfile",
        help="Generate CSV-file",
    )
    argparser.add_argument(
        "-m",
        "--csv-merge-jsons",
        type=int,
        default=None,
        const=-1,
        nargs="?",
        dest="merge",
        help="Merge JSONs into CSV (-1: auto, 0: all, 1: SP, 2: DP, 3: hidden)",
    )
    argparser.add_argument(
        "-M",
        "--csv-expand-jsons",
        type=str,
        default=None,
        nargs="?",
        dest="expand",
        help="Write JSONs of a CSV-file (destination: see -p)",
    )
    argparser.add_argument(
        "-x",
        "--csv-nogflops",
        action="store_true",
        default=False,
        dest="nogflops",
        help="Exclude real GFLOPS",
    )
    argparser.add_argument(
        "-p",
        "--jsons-dir",
        type=str,
        default=None,
        nargs="?",
        dest="jsondir",
        help="Directory to read/write JSONs (without shape: merge)",
    )
    argparser.add_argument(
        "-u",
        "--jsons-update",
        type=str,
        default="",
        nargs="?",
        dest="update",
        help="Update JSONs (device name optional)",
    )
    argparser.add_argument(
        "-c",
        "--check",
        type=float,
        default=0,
        nargs="?",
        help="Validate kernel (none:verify, epsilon - 0:off, -1:verify perf.)",
    )
    argparser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        default=False,
        help="Delete losing duplicates during merge (see --prefer)",
    )
    argparser.add_argument(
        "--prefer",
        type=str,
        default=None,
        choices=["fast", "new"],
        dest="prefer",
        help="Duplicate winner: fast (default) or new (without shape: merge)",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    argparser.add_argument(
        "-a",
        "--tuning-level",
        type=int,
        default=-1,
        nargs="?",
        dest="tlevel",
        help="Tunables: (0) all, (1) most, (2) some, (3) few, (4) least",
    )
    argparser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        default=False,
        help="Omit certain configurations",
    )
    argparser.add_argument(
        "-bm",
        "--initial-bm",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_BM", "0"),
        nargs="?",
        dest="bm",
        help="Block/tile size (0:auto)",
    )
    argparser.add_argument(
        "-bn",
        "--initial-bn",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_BN", "0"),
        nargs="?",
        dest="bn",
        help="Block/tile size (0:auto)",
    )
    argparser.add_argument(
        "-bk",
        "--initial-bk",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_BK", "0"),
        nargs="?",
        dest="bk",
        help="Block size (0:auto)",
    )
    argparser.add_argument(
        "-ws",
        "--initial-ws",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_WS", "0"),
        nargs="?",
        dest="ws",
        help="Minimum WG-size (0:auto)",
    )
    argparser.add_argument(
        "-wg",
        "--initial-wg",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_WG", "0"),
        dest="wg",
        help="Size of WG: subgroups (-1), tight (0), round-up (1), PoT (2)",
    )
    argparser.add_argument(
        "-lu",
        "--initial-lu",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_LU", "-1"),
        dest="lu",
        help="Loop unroll (-2) full, (-1) no hints (default),"
        + " (0) inner, (1) outer-dehint, (2) block-m",
    )
    argparser.add_argument(
        "-nz",
        "--initial-nz",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_NZ", "0"),
        dest="nz",
        help="Check (1) atomic increment to be non-zero (0:off)",
    )
    argparser.add_argument(
        "-al",
        "--initial-al",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_AL", "0"),
        dest="al",
        help="Access: transposed (0), linear (1)",
    )
    argparser.add_argument(
        "-tb",
        "--initial-tb",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_TB", "0"),
        dest="tb",
        help="Matrix B: untracked (0), tracked (1)",
    )
    argparser.add_argument(
        "-tc",
        "--initial-tc",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_TC", "1"),
        dest="tc",
        help="Matrix C: untracked (0), tracked (1)",
    )
    argparser.add_argument(
        "-ap",
        "--initial-ap",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_AP", "0"),
        dest="ap",
        help="Params: global (0), shared (1)",
    )
    argparser.add_argument(
        "-aa",
        "--initial-aa",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_AA", "0"),
        dest="aa",
        help="Matrix A: global (0), shared (1), register (2)",
    )
    argparser.add_argument(
        "-ab",
        "--initial-ab",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_AB", "0"),
        dest="ab",
        help="Matrix B: global (0), shared (1), register (2)",
    )
    argparser.add_argument(
        "-ac",
        "--initial-ac",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_AC", "0"),
        dest="ac",
        help="Matrix C: register (0), shared (1)",
    )
    argparser.add_argument(
        "-bs",
        "--initial-bs",
        type=int,
        default=env_intvalue("OPENCL_LIBSMM_SMM_BS", "0"),
        nargs="?",
        dest="bs",
        help="Minibatch size (0:auto)",
    )
    argparser.add_argument(
        "-mb",
        "--max-bs",
        type=int,
        default=0,
        nargs="?",
        dest="mb",
        help="Maximum (mini-)batch size (0:auto)",
    )
    argparser.add_argument(
        "-s",
        "--batchsize",
        type=int,
        default=0,
        nargs="?",
        dest="size",
        help="Size of batch (a.k.a. stacksize)",
    )
    argparser.add_argument(
        "--timeout",
        type=float,
        default=0,
        nargs="?",
        dest="timeout",
        help="Per-kernel timeout in seconds (0:unlimited)",
    )
    args, argd = argparser.parse_args(), argparser.parse_args([])
    # without a shape, a given JSON-directory or preference asks for a merge
    if args.mnk is None or os.path.isdir(args.mnk):
        if args.mnk is not None:
            args.jsondir = args.mnk
        given = args.jsondir is not None or args.prefer is not None
        if given and args.merge is None:
            args.merge = -1
        args.mnk = default_mnk
    if args.jsondir is None:
        args.jsondir = "."
    if args.prefer is None:
        args.prefer = "fast"
    # expanding a CSV-file needs neither a device nor the benchmark
    if args.expand is not None and "" != args.expand:
        if not os.path.isfile(args.expand):
            sys.tracebacklimit = 0
            raise RuntimeError("Cannot read {}.".format(args.expand))
        os.makedirs(args.jsondir, exist_ok=True)
        exit(csv_expand(args.expand, args.jsondir, args.csvsep))
    # OPENCL_LIBSMM_SMM_xx=tune|enabled|on must be given to permit tuning)
    if os.getenv("OPENCL_LIBSMM_SMM_WS") not in default_enable_tune:
        os.environ["OPENCL_LIBSMM_SMM_WS"] = "{}".format(args.ws)
    # fix tunables according to level of tuning
    if 1 <= args.tlevel or 0 > args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_BM"] = "{}".format(args.bm)
        os.environ["OPENCL_LIBSMM_SMM_BN"] = "{}".format(args.bn)
        os.environ["OPENCL_LIBSMM_SMM_AL"] = "{}".format(args.al)
    if 2 <= args.tlevel or 0 > args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_TB"] = "{}".format(args.tb)
        os.environ["OPENCL_LIBSMM_SMM_TC"] = "{}".format(args.tc)
        os.environ["OPENCL_LIBSMM_SMM_AP"] = "{}".format(args.ap)
        os.environ["OPENCL_LIBSMM_SMM_AC"] = "{}".format(args.ac)
        os.environ["OPENCL_LIBSMM_SMM_NZ"] = "{}".format(args.nz)
    if 3 <= args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_BK"] = "{}".format(args.bk)
        os.environ["OPENCL_LIBSMM_SMM_WG"] = "{}".format(args.wg)
    if 4 <= args.tlevel:
        os.environ["OPENCL_LIBSMM_SMM_LU"] = "{}".format(args.lu)
    if 0 == args.mb:
        args.mb = 64
    # construct and start tuner instance
    if os.path.isfile(args.mnk):
        with open(args.mnk, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                # comments carry a plan's annotations and its progress
                args.mnk = re.sub(r"\s+", "", line.split("#")[0])
                args.label = ""
                if args.mnk:
                    start(args)
                    print("")
    else:
        try:
            mnk = tuple(max(int(i), 1) for i in args.mnk.split("x"))
        except Exception:
            mnk = None
            pass
        if not mnk:
            sys.tracebacklimit = 0
            raise RuntimeError("Cannot parse MxNxK triplet or filename.")
        start(args)
