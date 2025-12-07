import time
import csv
import os
from statistics import mean, pstdev
from scapy.all import sniff, IP, TCP, UDP

# CONFIG
FLOW_TIMEOUT = 60
OUTPUT_CSV = "flows.csv"              # KMeans
DBSCAN_CSV = "dbscan_flows.csv"       # DBSCAN
INTERFACE = "ens34"                   # network interface

COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp",
    "flow_duration",
    "flow_byts_s", "flow_pkts_s",
    "fwd_pkts_s", "bwd_pkts_s",
    "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_pkts", "totlen_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std",
    "fwd_iat_mean", "fwd_iat_max", "fwd_iat_min", "fwd_iat_std",
    "bwd_iat_mean", "bwd_iat_max", "bwd_iat_min", "bwd_iat_std",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
]

DBSCAN_COLUMNS = [
    "timestamp",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "IN_BYTES",
    "OUT_BYTES",
    "FLOW_DURATION_MILLISECONDS",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

def safe_mean(v):
    return mean(v) if v else 0.0

def safe_std(v):
    return pstdev(v) if len(v) > 1 else 0.0


class Flow:
    def __init__(self, sip, dip, sport, dport, proto, ts, length, direction):
        self.src_ip = sip
        self.dst_ip = dip
        self.src_port = sport
        self.dst_port = dport
        self.proto = proto

        self.start_ts = ts
        self.last_ts = ts

        self.all_times = [ts]
        self.fwd_times = [ts] if direction == "fwd" else []
        self.bwd_times = [ts] if direction == "bwd" else []

        self.fwd_lens = [length] if direction == "fwd" else []
        self.bwd_lens = [length] if direction == "bwd" else []

        self.tot_fwd_pkts = 1 if direction == "fwd" else 0
        self.tot_bwd_pkts = 1 if direction == "bwd" else 0
        self.totlen_fwd_pkts = length if direction == "fwd" else 0
        self.totlen_bwd_pkts = length if direction == "bwd" else 0

        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0
        self.fin_flag_cnt = 0
        self.syn_flag_cnt = 0
        self.rst_flag_cnt = 0

    def update(self, ts, length, direction, tcp_flags=None):
        self.last_ts = ts
        self.all_times.append(ts)

        if direction == "fwd":
            self.tot_fwd_pkts += 1
            self.totlen_fwd_pkts += length
            self.fwd_lens.append(length)
            self.fwd_times.append(ts)
        else:
            self.tot_bwd_pkts += 1
            self.totlen_bwd_pkts += length
            self.bwd_lens.append(length)
            self.bwd_times.append(ts)

        if tcp_flags is not None:
            fin = bool(tcp_flags & 0x01)
            syn = bool(tcp_flags & 0x02)
            rst = bool(tcp_flags & 0x04)
            psh = bool(tcp_flags & 0x08)
            urg = bool(tcp_flags & 0x20)

            if fin: self.fin_flag_cnt += 1
            if syn: self.syn_flag_cnt += 1
            if rst: self.rst_flag_cnt += 1

            if direction == "fwd":
                if psh: self.fwd_psh_flags += 1
                if urg: self.fwd_urg_flags += 1
            else:
                if psh: self.bwd_psh_flags += 1
                if urg: self.bwd_urg_flags += 1

    def _iat_stats(self):

        def compute(arr):
            if len(arr) < 2:
                return (0,0,0,0)
            diffs = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
            return safe_mean(diffs), max(diffs), min(diffs), safe_std(diffs)

        flow = compute(self.all_times)
        fwd = compute(self.fwd_times)
        bwd = compute(self.bwd_times)
        return flow + fwd + bwd

    def to_row(self):
        duration = max(self.last_ts - self.start_ts, 1e-9)

        total_pkts = self.tot_fwd_pkts + self.tot_bwd_pkts
        total_bytes = self.totlen_fwd_pkts + self.totlen_bwd_pkts

        flow_byts_s = total_bytes / duration
        flow_pkts_s = total_pkts / duration

        fwd_pkts_s = self.tot_fwd_pkts / duration
        bwd_pkts_s = self.tot_bwd_pkts / duration

        all_lens = self.fwd_lens + self.bwd_lens

        if all_lens:
            pkt_len_max = max(all_lens)
            pkt_len_min = min(all_lens)
            pkt_len_mean = safe_mean(all_lens)
            pkt_len_std = safe_std(all_lens)
            pkt_len_var = pkt_len_std * pkt_len_std
        else:
            pkt_len_max = pkt_len_min = pkt_len_mean = pkt_len_std = pkt_len_var = 0

        def lens(arr):
            return (
                max(arr) if arr else 0,
                min(arr) if arr else 0,
                safe_mean(arr) if arr else 0,
                safe_std(arr) if arr else 0,
            )

        fwd_stats = lens(self.fwd_lens)
        bwd_stats = lens(self.bwd_lens)

        (
            flow_iat_mean, flow_iat_max, flow_iat_min, flow_iat_std,
            fwd_iat_mean, fwd_iat_max, fwd_iat_min, fwd_iat_std,
            bwd_iat_mean, bwd_iat_max, bwd_iat_min, bwd_iat_std,
        ) = self._iat_stats()

        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_ts))

        return [
            self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto, ts_str,
            duration,
            flow_byts_s, flow_pkts_s,
            fwd_pkts_s, bwd_pkts_s,
            self.tot_fwd_pkts, self.tot_bwd_pkts,
            self.totlen_fwd_pkts, self.totlen_bwd_pkts,
            *fwd_stats, *bwd_stats,
            pkt_len_max, pkt_len_min, pkt_len_mean, pkt_len_std, pkt_len_var,
            flow_iat_mean, flow_iat_max, flow_iat_min, flow_iat_std,
            fwd_iat_mean, fwd_iat_max, fwd_iat_min, fwd_iat_std,
            bwd_iat_mean, bwd_iat_max, bwd_iat_min, bwd_iat_std,
            self.fwd_psh_flags, self.bwd_psh_flags,
            self.fwd_urg_flags, self.bwd_urg_flags,
            self.fin_flag_cnt, self.syn_flag_cnt, self.rst_flag_cnt,
        ]


def write_dbscan_flow(flow: Flow):

    is_new = not os.path.exists(DBSCAN_CSV)

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(flow.last_ts))

    duration_ms = (flow.last_ts - flow.start_ts) * 1000
    duration_ms = max(duration_ms, 1)

    IN_BYTES = flow.totlen_bwd_pkts
    OUT_BYTES = flow.totlen_fwd_pkts

    SRC_TO_DST_SECOND_BYTES = OUT_BYTES / (duration_ms / 1000)
    DST_TO_SRC_SECOND_BYTES = IN_BYTES / (duration_ms / 1000)

    (
        _f_mean, _f_max, _f_min, _f_std,
        src_iat_avg, _, _, src_iat_std,
        dst_iat_avg, _, _, dst_iat_std
    ) = flow._iat_stats()

    row = {
        "timestamp": ts,
        "IPV4_SRC_ADDR": flow.src_ip,
        "IPV4_DST_ADDR": flow.dst_ip,
        "L4_DST_PORT": flow.dst_port,
        "IN_BYTES": IN_BYTES,
        "OUT_BYTES": OUT_BYTES,
        "FLOW_DURATION_MILLISECONDS": duration_ms,
        "SRC_TO_DST_SECOND_BYTES": SRC_TO_DST_SECOND_BYTES,
        "DST_TO_SRC_SECOND_BYTES": DST_TO_SRC_SECOND_BYTES,
        "SRC_TO_DST_IAT_AVG": src_iat_avg * 1000,
        "SRC_TO_DST_IAT_STDDEV": src_iat_std * 1000,
        "DST_TO_SRC_IAT_AVG": dst_iat_avg * 1000,
        "DST_TO_SRC_IAT_STDDEV": dst_iat_std * 1000,
    }

    with open(DBSCAN_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DBSCAN_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)


flows = {}

def get_flow_key(pkt):
    if IP not in pkt:
        return None, None
    ip = pkt[IP]
    proto = ip.proto

    src = ip.src
    dst = ip.dst
    sport = dport = 0

    if proto == 6 and TCP in pkt:
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif proto == 17 and UDP in pkt:
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport

    return (src, dst, sport, dport, proto), (dst, src, dport, sport, proto)

def write_flow(flow: Flow):
    # KMEANS
    is_new = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(COLUMNS)
        w.writerow(flow.to_row())

    # DBSCAN
    write_dbscan_flow(flow)


def expire_flows(now):
    dead = []
    for key, fl in flows.items():
        if now - fl.last_ts > FLOW_TIMEOUT:
            write_flow(fl)
            dead.append(key)
    for key in dead:
        del flows[key]


def process_packet(pkt):
    ts = float(pkt.time)
    expire_flows(ts)

    key_fwd, key_bwd = get_flow_key(pkt)
    if key_fwd is None:
        return

    length = len(pkt)

    if key_fwd in flows:
        direction = "fwd"
        flow = flows[key_fwd]
    elif key_bwd in flows:
        direction = "bwd"
        flow = flows[key_bwd]
    else:
        src, dst, sport, dport, proto = key_fwd
        flow = Flow(src, dst, sport, dport, proto, ts, length, "fwd")
        flows[key_fwd] = flow
        direction = "fwd"

    tcp_flags = int(pkt[TCP].flags) if TCP in pkt else None
    flow.update(ts, length, direction, tcp_flags)


def main():
    print(f"[+] Sniffing on {INTERFACE}, writing flows.csv and dbscan_flows.csv ...")
    sniff(iface=INTERFACE, prn=process_packet, store=False)


if __name__ == "__main__":
    main()
