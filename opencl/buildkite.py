#!/usr/bin/env python3
import matplotlib.pyplot as plot
import statistics
import requests
import pickle
import json
import math
import sys
import re


def summary(values, i=None):
    v = values if i is None else [v[i] for v in values]
    return statistics.geometric_mean([v for v in v if 0 < v]) if v else 0


argc = len(sys.argv)
if 2 < argc:
    url = sys.argv[1]
    auth_token = sys.argv[2]
else:
    sys.tracebacklimit = 0
    raise ValueError(sys.argv[0] + ": please pass URL and TOKEN!")

auth_param = {"Authorization": "Bearer {}".format(auth_token)}
params = {"per_page": 100, "page": 1}
devnames = None
metric = None
steps = None

# first name in sublists is used in plot
if not devnames:  # populate devnames
    with open("buildkite-devs.json", "r") as file:
        devnames = json.load(file)
else:  # e.g., [["GPU1", "host1", "host2"], ["GPU2", "host3"]]
    with open("buildkite-devs.json", "w") as file:
        json.dump(devnames, file, sort_keys=True, indent=2)
        file.write("\n")  # append newline at EOF

if not metric:  # populate device metrics
    with open("buildkite-metric.json", "r") as file:
        metric = json.load(file)
else:  # e.g., {"GPU1": [SPflpGFs, mbwGBs], "GPU2": [SPflpGFs, mbwGBs]}
    with open("buildkite-metric.json", "w") as file:
        json.dump(metric, file, sort_keys=True, indent=2)
        file.write("\n")  # append newline at EOF

if not steps:  # populate steps
    with open("buildkite-steps.json", "r") as file:
        steps = json.load(file)
else:  # e.g., {"Stepname1": {}, "Stepname2": {}}
    with open("buildkite-steps.json", "w") as file:
        json.dump(steps, file, sort_keys=True, indent=2)
        file.write("\n")  # append newline at EOF

cached = int(sys.argv[3]) if 3 < argc else 0
index = int(sys.argv[4]) if 4 < argc else -1
focus = sys.argv[5] if 5 < argc else "23x23x23"
lastn = int(sys.argv[6]) if 6 < argc else 5
sprat = max(float(sys.argv[7]) if 7 < argc else 2, 0)
maxn = max(int(sys.argv[8]) if 8 < argc else 0, 0)
maxm = max(int(sys.argv[9]) if 9 < argc else 0, 0)
vmin = min(max(int(sys.argv[10]) if 10 < argc else 0, 0), 100)
vmax = min(max(int(sys.argv[11]) if 11 < argc else 100, 0), 200)

try:  # determine last build already processed
    if 0 == cached:
        with open("buildkite-{}.last".format(focus), "r") as file:
            cached = int(file.read())
    if 0 < cached:
        with open("buildkite-{}.json".format(focus), "r") as file:
            steps.update(json.load(file))
except:  # noqa: E722
    cached = 0  # reset
    pass

work = 0
latest = 0
nbuild = cached
errsteps = errvals = 0
fmnk = tuple(map(int, focus.split("x")))
fai = (2 * fmnk[0] * fmnk[1] * fmnk[2]) / (
    4  # size [Byte] of an SP-element
    * sprat  # assume size-ratio is same as flops-ratio
    # assume C is hot in cache/registers (no RFO/RMW, etc)
    * (fmnk[2] * (fmnk[0] + fmnk[1]))
)
try:  # proceeed with cached results in case of an error
    builds = requests.get(url, params=params, headers=auth_param).json()
except:  # noqa: E722
    print("WARNING: failed to connect to {}\n".format(url))
    builds = None
    pass
while builds:
    for build in builds:
        nbuild = int(build["number"])
        if cached >= nbuild:
            break
        build_steps = [
            step for step in build["jobs"] if step["name"] in steps.keys()
        ]
        nsteps = 0
        try:
            for step in (
                step for step in build_steps if 0 == step["exit_status"]
            ):
                log = requests.get(step["log_url"], headers=auth_param)
                txt = log.text.replace("\\n", "\n")
                match = re.search(
                    'INFO ACC/OpenCL:\\s+ndevices=[0-9]+\\s+device[0-9]+="(.+)"',
                    txt,
                )
                if match and match.group(1):
                    devlog = match.group(1)
                else:
                    match = re.search("hostname:\\s+([\\w-]+)", txt)
                    devlog = match.group(1) if match and match.group(1) else ""
                if "" == devlog:
                    continue
                device = ""
                for devs in devnames:
                    if any(
                        re.search(dev, devlog, re.IGNORECASE) for dev in devs
                    ):
                        device = devs[0]
                if "" == device:
                    continue
                try:
                    mnklst = [
                        (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        for m in re.finditer(
                            "acc_bench_smm [0-9]+ [0-9]+ ([0-9]+) ([0-9]+) ([0-9]+) [0-9]+ [0-9]+",
                            txt,
                        )
                        if m and m.group(1) and m.group(2) and m.group(3)
                    ]
                    values = [
                        float(m.group(1))
                        for m in re.finditer(
                            "device: .+ ms\\s+(.+) GFLOPS/s", txt
                        )
                        if m and m.group(1)
                    ]
                    size_mnklst, size_values = len(mnklst), len(values)
                    if size_mnklst == size_values:
                        smlvals, medvals, bigvals, fvalue = [], [], [], None
                        for mnk, value in zip(mnklst, values):
                            if 0 < value:
                                size = math.prod(mnk)
                                if size <= (13**3):
                                    smlvals.append(value)
                                elif size <= (23**3):
                                    medvals.append(value)
                                else:
                                    bigvals.append(value)
                                if mnk == fmnk:
                                    fvalue = value
                        if fvalue and 0 < fvalue:
                            value = (
                                (round(summary(smlvals), 1), len(smlvals)),
                                (round(summary(medvals), 1), len(medvals)),
                                (round(summary(bigvals), 1), len(bigvals)),
                                fvalue,
                                nbuild,
                            )
                            if device in steps[step["name"]]:
                                if not any(
                                    nbuild == s[4]
                                    for s in steps[step["name"]][device]
                                ):
                                    steps[step["name"]][device].append(value)
                            else:
                                steps[step["name"]][device] = [value]
                            nsteps = nsteps + 1
                        else:
                            errvals = errvals + 1
                    else:
                        errvals = errvals + abs(size_mnklst - size_values)
                except:  # noqa: E722
                    errvals = errvals + 1
                    pass
        except:  # noqa: E722
            errsteps = errsteps + 1
            pass
        if nsteps == len(build_steps) and latest < nbuild:
            latest = nbuild
        print(".", end="", flush=True)
        work = work + 1
    if cached < nbuild:  # continue requesting pages
        params["page"] = params["page"] + 1  # next page
        builds = requests.get(url, params=params, headers=auth_param).json()
    else:  # terminate outer loop
        builds = []  # terminator

if 0 < work:  # there was some progress printed
    print("")
    if 0 < errsteps or 0 < errvals:
        print("Errors: steps={} vals={}".format(errsteps, errvals))

if 0 < latest:  # there was new data discovered
    for g in steps:  # order data from oldest to latest build
        for s in steps[g]:
            steps[g][s].sort(key=lambda v: v[4])
    with open("buildkite-{}.json".format(focus), "w") as file:
        json.dump(steps, file, sort_keys=True, indent=2)
    with open("buildkite-{}.last".format(focus), "w") as file:
        file.write("{}\n".format(latest))

# filter and rebuild collected results
filtered_steps = {}
for g in steps:
    step = {}
    for s in steps[g]:
        values = [
            v
            for v in steps[g][s]
            if 0 < sum(list(zip(*v[:3]))[1])
            and ((fai * metric[s][1] * vmin) <= (v[3] * 100))
            and ((fai * metric[s][1] * vmax) >= (v[3] * 100))
        ]
        step[s] = values
    filtered_steps[g] = step
steps = filtered_steps

mineff = maxeff = 0
for devs in devnames:
    minlen = min(
        len(step[devs[0]]) if devs[0] in step else 0 for step in steps.values()
    )
    if 0 < minlen:
        maxlen = max(len(step[devs[0]]) for step in steps.values())
        maxlen = min(maxlen, maxn if 0 < maxn else maxlen)
        d = maxlen - minlen
        main, (allaxs, focaxs) = plot.subplots(
            2, sharex=True, figsize=(9, 6), dpi=300
        )
        allaxs.get_xaxis().set_visible(False)
        focaxs.get_xaxis().set_visible(False)
        allaxs.set_ylabel("GFLOPS/s")
        focaxs.set_ylabel("GFLOPS/s")
        figures = {}
        stepeff = 0
        for step in steps:
            s = steps[step][devs[0]][::-1]
            g = s[: maxn if 0 < maxn else maxlen]
            t = g[: maxlen - (min(maxm, d) if len(g) == maxlen else 0)]
            if 0 > index or 2 < index:
                lst = [
                    (
                        math.exp(
                            (
                                (
                                    (math.log(v[0][0]) * v[0][1])
                                    if 0 < v[0][1]
                                    else 0
                                )
                                + (
                                    (math.log(v[1][0]) * v[1][1])
                                    if 0 < v[1][1]
                                    else 0
                                )
                                + (
                                    (math.log(v[2][0]) * v[2][1])
                                    if 0 < v[2][1]
                                    else 0
                                )
                            )
                            / sum(list(zip(*v[:3]))[1])
                        ),
                        v[3],
                    )
                    for v in t
                ]
            else:
                lst = [(v[index][0], v[3]) for v in t]
            val = list(zip(*t))
            foc = list(val[3])
            pea = fai * metric[devs[0]][1]
            eff = [v / pea for v in foc]
            all = list(list(zip(*lst))[0])
            pad = [None] * (maxlen - len(t))
            caption = step.split()
            runtime = caption[0]
            subject = caption[1]
            props = " ".join(caption[2:])
            laball = "{}{}..{}..{}={} GFLOPS/s".format(
                "last{}=".format(lastn) if 0 < lastn else "",
                round(
                    summary(val[0][:lastn], 0)
                    if 0 < lastn
                    else summary(val[0], 0)
                ),
                round(
                    summary(val[1][:lastn], 0)
                    if 0 < lastn
                    else summary(val[1], 0)
                ),
                round(
                    summary(val[2][:lastn], 0)
                    if 0 < lastn
                    else summary(val[2], 0)
                ),
                round(summary(all[:lastn]) if 0 < lastn else summary(all)),
            )
            efflast = round(
                100 * (summary(eff[:lastn]) if 0 < lastn else summary(eff))
            )
            labfoc = "{}{} GFLOPS/s ({}%)".format(
                "last{}=".format(lastn) if 0 < lastn else "",
                round(
                    summary(val[3][:lastn]) if 0 < lastn else summary(val[3])
                ),
                efflast,
            )
            fname = "buildkite-{}-{}.png".format(
                devs[0].lower(),
                re.sub(
                    r"(?u)[^-\w.]", "", props.strip().replace(" ", "_")
                ).lower(),
            )
            count = (
                max(list(zip(*val[0]))[1]),
                max(list(zip(*val[1]))[1]),
                max(list(zip(*val[2]))[1]),
            )
            if stepeff < efflast:
                stepeff = efflast
            if fname not in figures:
                figures[fname] = pickle.loads(pickle.dumps(main))
            figure = figures[fname]
            figure.suptitle(
                "Performance of {} {} on {}".format(subject, props, devs[0]),
                fontsize="x-large",
            )
            (allaxs, focaxs) = figure.axes
            allaxs.set_title("Summary of {} SMM-kernels".format(sum(count)))
            allaxs.plot(
                all + pad, ".:", label="{}: {}".format(runtime, laball)
            )
            focaxs.plot(
                foc + pad,
                ".:",
                label="{}: build={}..{} {}".format(
                    runtime, t[-1][4], t[0][4], labfoc
                ),
            )
            focaxs.set_title(
                "Single Kernel: MNK={}, AI={} FLOPS/Byte, Roofline={} GFLOPS/s".format(
                    focus, round(fai, 3), round(pea)
                )
            )
            print(
                "{} {} {}: {} (nbuilds={})".format(
                    devs[0], runtime, props, laball, len(t)
                )
            )
        if mineff > stepeff or 0 == mineff:
            mineff = stepeff
        if maxeff < stepeff:
            maxeff = stepeff
        for fname in figures:
            figure = figures[fname]
            (allaxs, focaxs) = figure.axes
            allaxs.legend()
            focaxs.legend()
            figure.gca().invert_xaxis()
            figure.tight_layout()
            figure.savefig(fname)
print(
    "Common build number (lockstep): {}\n".format(
        "{}->{}".format(cached, latest) if 0 < latest else cached
    )
)
for device in metric:
    peak = fai * metric[device][1] / 100  # perecentage
    print(
        "{}: {}..{} GFLOPS/s".format(
            device, round(mineff * peak), round(maxeff * peak)
        )
    )
